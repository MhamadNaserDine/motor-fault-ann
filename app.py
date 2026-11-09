"""
app.py -- Streamlit GUI
=======================
ANN-based Fault Detection & Diagnosis of Induction Motors

    streamlit run app.py

Pages
  1. Motor simulation        – dq model, fault injection, currents / speed / torque / spectrum
  2. Dataset generation      – many simulations -> MCSA features, class distribution
  3. Training                – choose network + hyper-parameters, live training/test error
  4. Results & comparison    – metrics table, confusion matrices, MLP vs RBF
  5. Live diagnosis demo     – unknown motor -> trained networks -> predicted condition
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from motor_model import MotorParams, simulate
from faults import CLASSES, SHORT, ICONS, SEVERITY_LABELS, make_fault_state, fault_frequencies
from features import FEATURE_NAMES, add_noise, extract_features
from dataset import generate_dataset
from networks import MODEL_NAMES, build_model
import plots
import train as T

st.set_page_config(page_title="Motor Fault ANN", page_icon="⚙️", layout="wide")
ROOT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ look & feel
_CSS = """
<style>
/* hand cursor on everything clickable */
div[data-baseweb="select"], div[data-baseweb="select"] *, ul[role="listbox"] li, li[role="option"],
div[role="option"], div[role="radiogroup"] label, label[data-baseweb="radio"], label[data-baseweb="checkbox"],
div[role="slider"], button, [data-testid="stExpander"] summary { cursor: pointer !important; }

/* hide the Streamlit "Deploy" button and footer */
[data-testid="stAppDeployButton"], .stDeployButton, footer { display: none !important; }

/* page header */
.page-head { border-left: 5px solid #2a9d8f; padding: 4px 0 4px 16px; margin: 0 0 18px 0; }
.page-head .kicker { font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; color: #2a9d8f;
                     font-weight: 700; }
.page-head h1 { font-size: 2.0rem; margin: 2px 0 4px 0; padding: 0; line-height: 1.2; }
.page-head .goal { color: #9aa4b2; font-size: 1.0rem; }

/* metric cards */
[data-testid="stMetric"] { background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.09);
                           border-radius: 12px; padding: 12px 16px; }
[data-testid="stMetricValue"] { font-size: 1.7rem; }

/* figures as rounded cards */
[data-testid="stImage"] img, .stPyplot img { border-radius: 10px; }

/* overview: KPI tiles and pipeline */
.kpi { background: rgba(42,157,143,0.10); border: 1px solid rgba(42,157,143,0.35); border-radius: 14px;
       padding: 16px 18px; height: 100%; }
.kpi .v { font-size: 2.0rem; font-weight: 800; color: #3cc0ae; line-height: 1.1; }
.kpi .l { color: #c9d1d9; font-size: .92rem; margin-top: 4px; }
.flow { display: flex; flex-wrap: wrap; gap: 8px; align-items: stretch; margin: 6px 0 4px 0; }
.flow .step { flex: 1 1 130px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10);
              border-radius: 12px; padding: 10px 12px; }
.flow .step b { color: #3cc0ae; display: block; font-size: .95rem; }
.flow .step span { color: #9aa4b2; font-size: .82rem; }
.flow .arrow { align-self: center; color: #2a9d8f; font-size: 1.3rem; }
.concl { background: rgba(42,157,143,0.10); border-left: 4px solid #2a9d8f; border-radius: 8px;
         padding: 10px 14px; margin: 6px 0 14px 0; }
.sb-status { font-size: .85rem; line-height: 1.7; }
</style>
"""
if hasattr(st, "html"):          # recent Streamlit: st.markdown drops <style>, st.html injects it
    st.html(_CSS)
else:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(kicker, title, goal):
    """Consistent page title: small section label, big title, one-line goal of the page."""
    st.markdown(f'<div class="page-head"><div class="kicker">{kicker}</div><h1>{title}</h1>'
                f'<div class="goal">🎯 <b>Goal:</b> {goal}</div></div>', unsafe_allow_html=True)


def conclusion(text):
    st.markdown(f'<div class="concl">💡 {text}</div>', unsafe_allow_html=True)


MODEL_HELP = {
    "Perceptron": "Single layer, hard limiter, Δw = η(d − y)x. Only linearly separable problems.",
    "ADALINE": "Single linear neuron per class, LMS rule Δw = η e x on the linear output.",
    "MLP-BP": "1 hidden layer, classic back-propagation: δ_j = e_j φ'(v_j), δ_k = Σ δ_j w_jk φ'(v_k).",
    "MLP-momentum": "BP + momentum: Δw(n) = η δ y + α Δw(n−1).",
    "MLP-adaptive": "BP + momentum + adaptive η(n): η↑ when E decreases, η↓ (step rejected) when E increases.",
    "MLP-LM": "Levenberg-Marquardt: Δw = −(JᵀJ + μI)⁻¹ Jᵀe, μ↓ after success, μ↑ after failure.",
    "RBF": "Gaussian hidden layer, centres = random training vectors, σ = d/√(2M), w = G⁺ y*.",
}


# ------------------------------------------------------------------ helpers
def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


@st.cache_data(show_spinner=False)
def load_saved_dataset():
    p = os.path.join(ROOT, "data", "dataset.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_resource(show_spinner=False)
def load_saved_bundle(task):
    p = os.path.join(ROOT, "models", f"{task}_bundle.pkl")
    if not os.path.exists(p):
        return None
    try:
        return T.load_bundle(p)
    except Exception as e:   # e.g. pickle made with another numpy version -> just re-run train.py
        st.error(f"Could not load {p} ({e}). Run `python train.py` to re-create the models.")
        return None


def load_saved_results(task):
    p = os.path.join(ROOT, "results", f"{task}_results.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return pd.DataFrame(json.load(f)).set_index("key")


def get_dataset():
    if "df" not in st.session_state:
        st.session_state.df = load_saved_dataset()
    return st.session_state.df


def trained():
    return st.session_state.setdefault("trained", {"detection": {}, "diagnosis": {}})


def get_prep(task):
    df = get_dataset()
    key = f"prep_{task}_{id(df)}"
    if key not in st.session_state:
        st.session_state[key] = T.prepare(df, task, seed=0)
    return st.session_state[key]


def run_sim(cond, sev, load, noise, t_end=3.0, standstill=False, seed=None):
    rng = np.random.default_rng(seed)
    mp = MotorParams()
    fs = make_fault_state(cond, sev if cond else 0, mp, rng)
    sig = simulate(mp, fs, load=load, t_end=t_end, start_from_standstill=standstill)
    if noise > 0:
        sig = add_noise(sig, noise, rng)
    return sig, mp


PREFERRED = ("MLP-BP", "MLP-LM", "MLP-momentum")   # MLP-BP: best mean accuracy over 10 runs


def ann_verdict(x):
    """Ask the trained networks (saved by train.py, else trained in this session) to judge a motor.
    Returns None when no network is available."""
    def pick(task):
        b = load_saved_bundle(task)
        if b is None and trained()[task]:
            p = get_prep(task)
            b = {"models": trained()[task], "scaler": p["scaler"], "labels": p["labels"]}
        if b is None:
            return None, None
        key = next((k for k in PREFERRED if k in b["models"]), list(b["models"])[0])
        return b, key
    bg, kg = pick("diagnosis")
    if bg is None:
        return None
    pg = bg["models"][kg].predict_proba(bg["scaler"].transform(x[None, :]))[0]
    cls = int(np.argmax(pg))
    out = {"cls": cls, "conf": float(pg[cls]), "proba": pg, "model": bg["models"][kg].name}
    bd, kd = pick("detection")
    if bd is not None:
        pdx = bd["models"][kd].predict_proba(bd["scaler"].transform(x[None, :]))[0]
        out["p_faulty"] = float(pdx[1])
    out["faulty"] = cls != 0
    return out


def verdict_text(v):
    if v is None:
        return None
    if not v["faulty"]:
        return f"ANN VERDICT: HEALTHY  (confidence {v['conf']*100:.0f} %)"
    return f"ANN VERDICT: FAULTY – {CLASSES[v['cls']]}  (confidence {v['conf']*100:.0f} %)"


def symptom_checks(fd, sig):
    """Simple engineer's rules (thresholds = upper range of healthy motors in our dataset)."""
    speed_pp = float(np.ptp(sig["speed_rpm"]))
    rows = [
        ("Speed oscillation (peak-to-peak)", f"{speed_pp:.1f} rpm", "< 5 rpm", speed_pp > 5, "Broken bars"),
        ("BRB sidebands (1 ± 2s)f", f"{max(fd['brb_lsb_dB'], fd['brb_usb_dB']):.0f} dB", "< −38 dB",
         max(fd['brb_lsb_dB'], fd['brb_usb_dB']) > -38, "Broken bars"),
        ("Neg.-seq. current I₂/I₁", f"{fd['I2_I1_negseq']*100:.1f} %", "< 6 %", fd["I2_I1_negseq"] > 0.06,
         "Inter-turn short or supply unbalance"),
        ("Neg.-seq. voltage V₂/V₁", f"{fd['V2_V1_negseq']*100:.2f} %", "< 0.8 %", fd["V2_V1_negseq"] > 0.008,
         "Supply voltage unbalance"),
        ("Bearing components |f ± f_o|", f"{max(fd['brg_lsb_dB'], fd['brg_usb_dB']):.0f} dB", "< −62 dB",
         max(fd['brg_lsb_dB'], fd['brg_usb_dB']) > -62, "Bearing fault"),
    ]
    return pd.DataFrame([{"symptom": r[0], "measured": r[1], "healthy if": r[2],
                          "status": "⚠️ abnormal" if r[3] else "✅ normal", "points to": r[4] if r[3] else "—"}
                         for r in rows]).set_index("symptom")


# ------------------------------------------------------------------ sidebar
PAGES = ["0 · Overview", "1 · Motor simulation", "2 · Dataset generation", "3 · Training",
         "4 · Results & comparison", "5 · Experiments", "6 · Live diagnosis demo"]
ICON = {"0": "🏠", "1": "⚡", "2": "🗂️", "3": "🧠", "4": "📊", "5": "🔬", "6": "🩺"}
st.sidebar.markdown("## ⚙️ Motor Fault ANN")
st.sidebar.caption("Fault detection & diagnosis of induction motors with artificial neural networks\n\n"
                   "AI-Based Control Systems · USJ – ESIB")
page = st.sidebar.radio("Navigation", PAGES, format_func=lambda p: f"{ICON[p[0]]}  {p[4:]}")
_ds = load_saved_dataset()
_ok = lambda c: "🟢" if c else "🔴"
st.sidebar.markdown(
    f'<div class="sb-status">{_ok(_ds is not None)} Dataset: {len(_ds) if _ds is not None else "missing"} motors<br>'
    f'{_ok(os.path.exists(os.path.join(ROOT, "models", "diagnosis_bundle.pkl")))} Trained networks (train.py)<br>'
    f'{_ok(os.path.exists(os.path.join(ROOT, "results", "experiments.json")))} Experiments (experiments.py)</div>',
    unsafe_allow_html=True)
mp0 = MotorParams()
with st.sidebar.expander("Motor nameplate / parameters"):
    st.markdown(f"""
- **{mp0.P_rated/1e3:.1f} kW**, {mp0.V_ll:.0f} V (Y), {mp0.f_s:.0f} Hz, {2*mp0.p} poles
- R_s = {mp0.Rs} Ω, R_r' = {mp0.Rr} Ω
- L_ls = L_lr = {mp0.Lls*1e3:.1f} mH, L_m = {mp0.Lm*1e3:.0f} mH
- J = {mp0.J} kg·m², {mp0.n_bars} rotor bars
- Rated torque ≈ {mp0.T_rated:.1f} N·m, ≈ 1443 rpm
- Bearing: {mp0.n_balls} balls, BPFO = {mp0.bpfo_ratio:.2f} × f_r
""")

# ================================================================== 0. OVERVIEW
if page.startswith("0"):
    page_header("ANN · Fault detection & diagnosis", "Induction motor fault diagnosis with neural networks",
                "find motor faults early from the stator current only, before the motor breaks down.")
    res_d = load_saved_results("diagnosis")
    best_acc = f"{res_d.drop(index='sklearn-MLP', errors='ignore')['accuracy'].max()*100:.1f} %" if res_d is not None else "—"
    _pe = os.path.join(ROOT, "results", "experiments.json")
    if os.path.exists(_pe):          # prefer the mean over 10 independent runs (more honest than one run)
        _sm = pd.DataFrame(json.load(open(_pe)).get("repeated", {}).get("summary", []))
        if len(_sm):
            _d = _sm[_sm.task == "diagnosis"].set_index("model")
            _b = _d.acc_mean.idxmax()
            best_acc = f"{_d.loc[_b, 'acc_mean']*100:.1f} ± {_d.loc[_b, 'acc_std']*100:.1f} %"
    ds = get_dataset()
    k = st.columns(4)
    for col, v, l in zip(k, [best_acc, f"{len(ds) if ds is not None else 0}", "7", "5"],
                         ["best diagnosis accuracy (MLP-BP, mean of 10 runs)", "simulated motors in the dataset",
                          "neural networks compared (from scratch)", "motor conditions recognised"]):
        col.markdown(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
    st.write("")
    st.subheader("How it works")
    st.markdown("""<div class="flow">
<div class="step"><b>1 · Virtual motor</b><span>2.2 kW dq model solved with scipy</span></div><div class="arrow">➜</div>
<div class="step"><b>2 · Fault injection</b><span>broken bars, short circuit, bearing, unbalance</span></div><div class="arrow">➜</div>
<div class="step"><b>3 · Measurement</b><span>3 currents + voltages with sensor noise</span></div><div class="arrow">➜</div>
<div class="step"><b>4 · 19 features</b><span>FFT sidebands, unbalance, Park's vector…</span></div><div class="arrow">➜</div>
<div class="step"><b>5 · Neural network</b><span>Perceptron … MLP-LM, RBF</span></div><div class="arrow">➜</div>
<div class="step"><b>6 · Verdict</b><span>Healthy ✅ / which fault ⚠️ + confidence</span></div>
</div>""", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Motor conditions")
        st.table(pd.DataFrame({
            "condition": CLASSES,
            "sign in the current": ["clean, balanced 50 Hz current", "sidebands at (1 ± 2s)·f, 2sf speed ripple",
                                    "phase unbalance, negative-sequence current", "components at |f ± f_bearing|",
                                    "unbalanced voltages and currents"]}).set_index("condition"))
    with c2:
        st.subheader("The pages of this app")
        st.table(pd.DataFrame({
            "page": [f"{ICON[p[0]]} {p[4:]}" for p in PAGES[1:]],
            "goal": ["see what each fault looks like", "create the training data", "teach the networks",
                     "compare the 7 networks", "test reliability (noise, load, size, repeats)",
                     "diagnose a new, unknown motor"]}).set_index("page"))
    fig_cmp = os.path.join(ROOT, "figures", "diagnosis_comparison.png")
    if os.path.exists(fig_cmp):
        st.subheader("Main result")
        st.image(fig_cmp, caption="Validation-set metrics of the 7 networks (fault diagnosis, 5 classes)")

# ================================================================== 1. SIMULATION
elif page.startswith("1"):
    page_header("Step 1 · Motor simulation", "Induction motor simulation",
                "choose a motor condition, run the dq model and see its signals, features and the ANN verdict.")
    c1, c2, c3, c4 = st.columns(4)
    cond = c1.selectbox("Motor condition", range(len(CLASSES)), format_func=lambda i: CLASSES[i])
    sev = c2.select_slider("Fault severity", [1, 2, 3], value=2, disabled=cond == 0,
                           format_func=lambda s: SEVERITY_LABELS[cond].get(s, "—") if cond else "—")
    load = c3.slider("Load (% of rated torque)", 0, 120, 80, 5) / 100
    noise = c4.slider("Measurement noise (% of RMS)", 0.0, 5.0, 0.5, 0.1) / 100
    c5, c6, c7 = st.columns(3)
    t_end = c5.slider("Recorded time (s)", 1.0, 5.0, 3.0, 0.5)
    standstill = c6.checkbox("Direct-on-line start from standstill", False,
                             help="Shows the start-up transient (slower; spectrum then includes the transient).")
    seed = c7.number_input("Random seed", 0, 10_000, 1)

    if st.button("▶ Run simulation", type="primary") or "sim" not in st.session_state:
        with st.spinner("Solving the motor ODEs (scipy solve_ivp)..."):
            sig, mp = run_sim(cond, sev, load, noise, t_end, standstill, int(seed))
        st.session_state.sim = (sig, cond, sev, load)
    sig, cond_s, sev_s, load_s = st.session_state.sim
    f = extract_features(sig)
    fd = dict(zip(FEATURE_NAMES, f))
    # ---- verdict: is this motor healthy or not? ---------------------------
    v = ann_verdict(f)
    vt = verdict_text(v)
    truth = CLASSES[cond_s] + ("" if cond_s == 0 else f" – {SEVERITY_LABELS[cond_s][sev_s]}")
    if v is None:
        st.warning("No trained network found – run `python train.py` (or train on page 3) to get the ANN verdict. "
                   "The symptom checklist below still works.")
    else:
        msg = f"### {'FAULTY ⚠️ – ' + CLASSES[v['cls']] if v['faulty'] else 'HEALTHY ✅'}\n" \
              f"ANN verdict ({v['model']}), confidence **{v['conf']*100:.0f} %**"
        if "p_faulty" in v:
            msg += f" · detection network: probability of a fault **{v['p_faulty']*100:.0f} %**"
        (st.error if v["faulty"] else st.success)(msg)
        ok = "✔ the network is right" if v["cls"] == cond_s else "✘ the network is wrong"
        st.caption(f"Condition you simulated (ground truth): **{truth}**, load {load_s*100:.0f} % → {ok}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean speed", f"{sig['speed_rpm'].mean():.0f} rpm")
    m2.metric("Slip", f"{fd['slip']*100:.2f} %")
    m3.metric("Current (RMS)", f"{fd['I_rms']:.2f} A")
    m4, m5, _ = st.columns(3)
    m4.metric("I₂/I₁ (unbalance)", f"{fd['I2_I1_negseq']*100:.2f} %")
    m5.metric("Mean torque", f"{sig['torque'].mean():.2f} N·m")

    def with_verdict(fig, short=False):
        if v is None:
            return fig
        text = (("FAULTY: " + SHORT[v["cls"]]) if v["faulty"] else "HEALTHY") if short else vt
        return plots.add_verdict(fig, text, v["faulty"])

    show(with_verdict(plots.motor_signals(sig, f"Simulated: {truth} – load {load_s*100:.0f} %",
                                          t_show=min(0.2, sig['t'][-1]) if not standstill else sig["t"][-1])))
    fmax = st.slider("Spectrum: max frequency (Hz)", 60, 500, 200, 10)
    show(with_verdict(plots.current_spectrum(sig, cond_s if cond_s else None, fd["slip"], fmax=fmax,
                                             title=f"Phase-a current spectrum – simulated: {CLASSES[cond_s]}")))
    if cond_s:
        ff = fault_frequencies(cond_s, sig["f_s"], fd["slip"])
        st.caption("Theoretical fault frequencies (red dashed): " + ", ".join(f"{x:.2f} Hz" for x in ff))

    st.subheader("Symptom checklist (how an engineer would judge it by hand)")
    st.table(symptom_checks(fd, sig))
    c1, c2 = st.columns([1, 1.4])
    with c1:
        show(with_verdict(plots.park_vector({CLASSES[cond_s]: sig}), short=True))
    with c2:
        st.subheader("Extracted MCSA features")
        st.dataframe(pd.DataFrame({"feature": FEATURE_NAMES, "value": f}).set_index("feature"),
                     width="stretch", height=420)

# ================================================================== 2. DATASET
elif page.startswith("2"):
    page_header("Step 2 · Data", "Dataset generation",
                "create many simulated motors and turn each one into 19 numbers the networks can learn from.")
    st.markdown("Each sample = one simulated motor with a random **class, severity (1-3), load and noise**, "
                "± 5 % parameter spread, then 19 MCSA features. Split: **training 60 % / test 20 % / validation 20 %**.")
    c1, c2, c3, c4 = st.columns(4)
    n = c1.number_input("Samples per class", 10, 1000, 40, 10)
    noise = c2.slider("Noise level (% of RMS)", 0.0, 5.0, 1.0, 0.1) / 100
    lr = c3.slider("Load range (%)", 0, 120, (30, 110), 5)
    seed = c4.number_input("Seed", 0, 10_000, 0)
    ncpu = os.cpu_count() or 1
    jobs = st.slider("Parallel processes", 1, ncpu, 1, help="Uses several CPU cores. Keep 1 if you see errors "
                     "on Windows.") if ncpu > 1 else 1
    save_csv = st.checkbox("Also save the generated dataset as data/dataset.csv (used by train.py)", value=False)
    c1, c2 = st.columns(2)
    est = n * len(CLASSES) * 0.8 / jobs
    if c1.button(f"⚙ Generate {n*len(CLASSES)} simulations (≈ {est/60:.1f} min)", type="primary"):
        bar = st.progress(0.0, "simulating...")
        df = generate_dataset(int(n), noise, (lr[0] / 100, lr[1] / 100), int(seed), n_jobs=jobs,
                              progress=lambda i, N: bar.progress(i / N, f"simulated {i}/{N} motors"))
        st.session_state.df = df
        st.session_state.trained = {"detection": {}, "diagnosis": {}}
        if save_csv:
            df.to_csv(os.path.join(ROOT, "data", "dataset.csv"), index=False)
        st.success(f"Generated {len(df)} samples.")
    if c2.button("📂 Load the saved dataset (data/dataset.csv)"):
        load_saved_dataset.clear()
        st.session_state.df = load_saved_dataset()
        st.session_state.trained = {"detection": {}, "diagnosis": {}}

    df = get_dataset()
    if df is None:
        st.info("No dataset yet – generate one (or run `python train.py`).")
    else:
        st.subheader(f"Current dataset: {len(df)} samples, {len(FEATURE_NAMES)} features")
        c1, c2 = st.columns([1, 1.3])
        with c1:
            show(plots.class_distribution(df))
            sets = T.split_3sets(df)
            st.table(pd.DataFrame({"set": ["training", "test (early stopping)", "validation (final)"],
                                   "samples": [len(sets["train"]), len(sets["test"]), len(sets["val"])]})
                     .set_index("set"))
        with c2:
            st.dataframe(df[["class", "severity", "load"] + FEATURE_NAMES].round(4), height=420,
                         width="stretch")
        st.download_button("⬇ Download CSV", df.to_csv(index=False), "motor_dataset.csv", "text/csv")
        feats = st.multiselect("Features to compare", FEATURE_NAMES,
                               ["I2_I1_negseq", "V2_V1_negseq", "brb_lsb_dB", "brg_lsb_dB", "park_eccentricity",
                                "h3_dB"])
        if feats:
            show(plots.feature_boxplots(df, feats))

# ================================================================== 3. TRAINING
elif page.startswith("3"):
    page_header("Step 3 · Learning", "Training the neural networks",
                "teach a network to recognise the faults and watch the training / test error live.")
    df = get_dataset()
    if df is None:
        st.warning("Generate or load a dataset first (page 2).")
        st.stop()
    c1, c2 = st.columns(2)
    task = c1.radio("Task", ["diagnosis", "detection"], horizontal=True,
                    format_func=lambda t: "Fault diagnosis (5 classes)" if t == "diagnosis" else "Fault detection (healthy / faulty)")
    kind = c2.selectbox("Network", MODEL_NAMES, index=3)
    st.caption(MODEL_HELP[kind])
    d = T.DEFAULT_CFG
    c1, c2, c3, c4, c5 = st.columns(5)
    n_hidden = c1.number_input("Hidden neurons", 2, 100, d["n_hidden"], disabled=kind in ("Perceptron", "ADALINE", "RBF"))
    eta = c2.number_input("Learning rate η", 0.0001, 2.0, d["eta"], format="%.4f",
                          disabled=kind in ("MLP-LM", "RBF"))
    alpha = c3.number_input("Momentum α", 0.0, 0.99, d["alpha"], disabled=kind not in ("MLP-momentum", "MLP-adaptive"))
    mu = c4.number_input("LM μ (initial)", 1e-6, 100.0, d["mu"], format="%.4f", disabled=kind != "MLP-LM")
    epochs = c5.number_input("Epochs", 1, 3000, d["epochs"], disabled=kind == "RBF",
                             help="Levenberg-Marquardt uses epochs/3 iterations (each iteration is much more powerful).")
    c1, c2 = st.columns(2)
    n_centers = c1.number_input("RBF centres M", 2, 400, d["n_centers"], disabled=kind != "RBF")
    act = c2.selectbox("Hidden activation φ", ["tanh", "logistic"], disabled=not kind.startswith("MLP"))
    cfg = dict(n_hidden=int(n_hidden), eta=float(eta), alpha=float(alpha), mu=float(mu), epochs=int(epochs),
               n_centers=int(n_centers), hidden_act=act, seed=0)
    prep = get_prep(task)
    st.caption(f"training {len(prep['y_train'])} · test {len(prep['y_test'])} · validation {len(prep['y_val'])} samples")

    def train_with_live_plot(k):
        ph = st.empty()
        bar = st.progress(0.0)
        every = max(1, cfg["epochs"] // 25)
        n_ep = cfg["epochs"] if k not in ("MLP-LM",) else max(10, cfg["epochs"] // 3)

        def cb(model, ep):
            bar.progress(min(ep / n_ep, 1.0), f"{k}: epoch {ep}/{n_ep}")
            if ep % every == 0 or ep == n_ep:
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(model.history["epoch"], model.history["train_E"], label="training error", color=plots.COLORS[2])
                ax.plot(model.history["epoch"], model.history["test_E"], label="test error", color=plots.COLORS[1])
                ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("E = ½Σe²"); ax.legend()
                ax.set_title(f"{k} – live training")
                ph.pyplot(fig)
                plt.close(fig)
        m = T.train_one(k, prep, cfg, callback=cb)
        ph.empty()
        bar.empty()
        return m

    c1, c2 = st.columns(2)
    if c1.button(f"🧠 Train {kind}", type="primary"):
        m = train_with_live_plot(kind)
        trained()[task][kind] = m
        st.session_state.last = (task, kind)
    if c2.button("Train ALL 7 networks (for the comparison page)"):
        for k in MODEL_NAMES:
            trained()[task][k] = train_with_live_plot(k)
        st.session_state.last = (task, kind)
        st.success("All networks trained – see page 4.")

    if "last" in st.session_state and st.session_state.last[1] in trained()[st.session_state.last[0]]:
        t_, k_ = st.session_state.last
        p_ = get_prep(t_)
        m = trained()[t_][k_]
        r = T.result_row(m, p_)
        st.subheader(f"{m.name} – {t_}")
        a, b, c, d_, e = st.columns(5)
        a.metric("Validation accuracy", f"{r['accuracy']*100:.1f} %")
        b.metric("Precision", f"{r['precision']*100:.1f} %")
        c.metric("Recall", f"{r['recall']*100:.1f} %")
        d_.metric("F1", f"{r['f1']*100:.1f} %")
        e.metric("Training time", f"{r['train_time_s']:.2f} s", f"point A = epoch {m.best_epoch}", delta_color="off")
        c1, c2 = st.columns([1.4, 1])
        with c1:
            if k_ != "RBF":
                show(plots.learning_curve(m))
                if k_ in ("MLP-adaptive", "MLP-LM"):
                    show(plots.adaptive_eta(m, "η(n) during training" if k_ == "MLP-adaptive" else "1/μ during training"))
            else:
                st.info(f"RBF trained in one shot: M = {m.M} centres, d = {m.d:.2f}, σ = d/√(2M) = {m.sigma:.3f}, "
                        "w = G⁺y*  (no epochs).")
        with c2:
            labs = SHORT if t_ == "diagnosis" else ["Healthy", "Faulty"]
            show(plots.confusion(np.array(r["cm"]), labs, "Confusion matrix (validation set)"))
        if st.button("💾 Save the networks trained in this session (models/<task>_bundle_app.pkl)"):
            T.save_bundle(trained()[t_], p_, os.path.join(ROOT, "models", f"{t_}_bundle_app.pkl"))
            st.success("Saved.")

# ================================================================== 4. RESULTS
elif page.startswith("4"):
    page_header("Step 4 · Evaluation", "Results & comparison",
                "compare the 7 networks on the validation set and find the best one.")
    task = st.radio("Task", ["diagnosis", "detection"], horizontal=True)
    src = st.radio("Source", ["Networks trained in this session", "Saved results (python train.py)"], horizontal=True)
    table = None
    if src.startswith("Networks"):
        tr = trained()[task]
        if not tr:
            st.info("No network trained in this session for this task – use page 3, or choose 'Saved results'.")
        else:
            prep = get_prep(task)
            rows = []
            for k, m in tr.items():
                r = T.result_row(m, prep)
                r["key"] = k
                rows.append(r)
            table = pd.DataFrame(rows).set_index("key")
    else:
        table = load_saved_results(task)
        if table is None:
            st.info("No saved results – run `python train.py` first.")
    if table is not None:
        show_t = table[["model", "accuracy", "precision", "recall", "f1", "train_acc", "test_acc",
                        "train_time_s", "epochs_run", "best_epoch_A"]].copy()
        for c in ["accuracy", "precision", "recall", "f1", "train_acc", "test_acc"]:
            show_t[c] = (show_t[c] * 100).round(1)
        show_t["train_time_s"] = show_t["train_time_s"].round(3)
        st.dataframe(show_t, width="stretch")
        show(plots.comparison_bars(table, f"{task}: validation-set metrics"))
        show(plots.time_bars(table))
        labs = SHORT if task == "diagnosis" else ["Healthy", "Faulty"]
        show(plots.confusion_grid({k: r for k, r in table.iterrows()}, labs, "Confusion matrices (validation set)"))

    st.subheader("MLP vs RBF")
    st.markdown("""
| | **MLP (back-propagation / LM)** | **RBF (fixed centres + pseudo-inverse)** |
|---|---|---|
| Hidden neuron | sigmoid/tanh of a weighted sum → **global** response (a hyperplane cuts the whole input space) | Gaussian of a distance → **local** response (only near its centre) |
| Training | iterative, non-linear optimisation (gradient, LM); risk of local minima; needs η, α, epochs | centres & σ fixed, output weights by **one linear least-squares solve** w = G⁺y* → no local minima |
| Speed | seconds (BP) – fewer s (LM, but heavy Jacobian) | milliseconds |
| Size | few hidden neurons are enough | needs many centres to cover the input space |
| Extrapolation | smoother outside the training domain | output → 0 far from all centres (an unknown operating point gives low confidence) |
""")
    fig_dir = os.path.join(ROOT, "figures")
    if os.path.isdir(fig_dir):
        with st.expander("Saved report figures (figures/*.png)"):
            for fn in sorted(os.listdir(fig_dir)):
                if fn.endswith(".png"):
                    st.image(os.path.join(fig_dir, fn), caption=fn)

# ================================================================== 5. EXPERIMENTS
elif page.startswith("5"):
    page_header("Step 5 · Reliability", "Experiments",
                "check how reliable the networks are: with more noise, at other loads, for early faults, "
                "over many runs, and for different network sizes.")
    p_exp = os.path.join(ROOT, "results", "experiments.json")
    if not os.path.exists(p_exp):
        st.info("No experiment results yet – run `python experiments.py` (≈ 10 min).")
        st.stop()
    E = json.load(open(p_exp))
    tabs = st.tabs(["🌫️ Noise, load & severity", "🔁 Repeated runs", "🧩 Hidden neurons"])

    with tabs[0]:
        if "robustness" in E:
            R = E["robustness"]
            st.markdown(f"**{R['n_motors']} fresh motors** (never used for training) are diagnosed by the saved networks.")
            show(plots.noise_robustness_plot(R["noise"]))
            nz = pd.DataFrame(R["noise"]).pivot(index="model", columns="noise", values="accuracy")
            nz.columns = [f"{c*100:g} %" for c in nz.columns]
            st.dataframe((nz * 100).round(1).reindex([m for m in plots.MODEL_ORDER if m in nz.index]), width="stretch")
            nzt = pd.DataFrame(R["noise"]).pivot(index="model", columns="noise", values="accuracy")
            mlp_in = nzt.loc["MLP-BP", [c for c in nzt.columns if c <= 0.02]].min()
            conclusion(f"Inside the noise range used for training (≤ 2 %) the MLPs stay at {mlp_in*100:.0f} % or more. "
                       f"Outside it they degrade fast: at 5 % noise MLP-BP falls to {nzt.loc['MLP-BP', 0.05]*100:.0f} %, "
                       f"while the simpler ADALINE keeps {nzt.loc['ADALINE', 0.05]*100:.0f} %. The MLP learned sharp "
                       "boundaries tuned to the training conditions, so it must be trained with the noise level of the "
                       "real sensors (or a wider range).")
            c1, c2 = st.columns(2)
            with c1:
                show(plots.breakdown_plot(R["severity"], "severity", {0: "Healthy", 1: "Sev. 1 (early)", 2: "Sev. 2",
                                                                     3: "Sev. 3 (severe)"}, "Accuracy vs fault severity"))
            with c2:
                show(plots.breakdown_plot(R["load"], "load", {}, "Accuracy vs load"))
            sv = pd.DataFrame(R["severity"]).query("model == 'MLP-BP'").set_index("severity")["accuracy"]
            ld = pd.DataFrame(R["load"]).query("model == 'MLP-BP'").set_index("load")["accuracy"]
            conclusion(f"MLP-BP recognises early faults (severity 1) {sv.loc[1]*100:.0f} % of the time and medium/severe "
                       f"faults {min(sv.loc[2], sv.loc[3])*100:.0f}" + (f"–{max(sv.loc[2], sv.loc[3])*100:.0f}" if round(sv.loc[2], 2) != round(sv.loc[3], 2) else "") + " %: the small faults "
                       f"cause almost all the errors. Across loads it stays between {ld.min()*100:.0f} % "
                       f"({ld.idxmin()}) and {ld.max()*100:.0f} % ({ld.idxmax()}).")

    with tabs[1]:
        if "repeated" in E:
            Rr = E["repeated"]
            st.markdown(f"Each network was trained **{Rr['n_runs']} times**, each time with a new random "
                        "training / test / validation split and new initial weights.")
            for task in ("diagnosis", "detection"):
                show(plots.repeated_runs_box(Rr["runs"], task))
            sm = pd.DataFrame(Rr["summary"])
            tbl = sm.assign(**{"accuracy (mean ± std)": sm.apply(lambda r: f"{r.acc_mean*100:.1f} ± {r.acc_std*100:.1f} %", axis=1),
                               "min – max": sm.apply(lambda r: f"{r.acc_min*100:.1f} – {r.acc_max*100:.1f} %", axis=1),
                               "time (s)": sm.time_mean.round(2)})
            st.dataframe(tbl[["task", "model", "accuracy (mean ± std)", "min – max", "time (s)"]], width="stretch",
                         hide_index=True)
            d = sm[sm.task == "diagnosis"].set_index("model")
            best = d.acc_mean.idxmax()
            conclusion(f"Diagnosis: {best} is the best on average ({d.loc[best,'acc_mean']*100:.1f} ± "
                       f"{d.loc[best,'acc_std']*100:.1f} %). The MLPs are clearly better than ADALINE "
                       f"({d.loc['ADALINE','acc_mean']*100:.1f} %) and RBF ({d.loc['RBF','acc_mean']*100:.1f} %), but only "
                       f"slightly better than the Perceptron ({d.loc['Perceptron','acc_mean']*100:.1f} %): the difference "
                       f"is smaller than the run-to-run spread, because the 19 features already make the classes almost "
                       f"linearly separable. A single run can therefore exaggerate differences.")

    with tabs[2]:
        if "hidden" in E:
            H = E["hidden"]
            st.markdown(f"MLP (BP + momentum) trained with different numbers of hidden neurons, "
                        f"{H['n_seeds']} runs each.")
            show(plots.hidden_sweep_plot(H["rows"]))
            hd = pd.DataFrame(H["rows"]).groupby("H").mean(numeric_only=True)
            plateau = int(hd.index[(hd.val_acc >= hd.val_acc.max() - 0.01)].min())
            conclusion(f"With 1–2 hidden neurons the network is too small (under-fitting: {hd.loc[1,'val_acc']*100:.0f} % and "
                       f"{hd.loc[2,'val_acc']*100:.0f} %). From about H = {plateau} the validation accuracy reaches a plateau "
                       f"(≈ {hd.loc[plateau:,'val_acc'].mean()*100:.0f} %): more neurons do not help. The training error keeps "
                       f"falling with H, while the test error at the end of training rises slightly (over-fitting), so early "
                       f"stopping at point A is what keeps the big networks accurate.")


# ================================================================== 6. LIVE DEMO
else:
    page_header("Step 6 · Use", "Live diagnosis demo",
                "test the trained network on a new motor and check whether its answer is right.")
    src = st.radio("Networks", ["Saved (python train.py)", "Trained in this session"], horizontal=True)

    def bundle(task):
        if src.startswith("Saved"):
            return load_saved_bundle(task)
        tr = trained()[task]
        if not tr:
            return None
        p = get_prep(task)
        return {"models": tr, "scaler": p["scaler"], "labels": p["labels"]}

    bd, bg = bundle("detection"), bundle("diagnosis")
    if bg is None:
        st.warning("No trained diagnosis network available – run `python train.py` or train on page 3.")
        st.stop()
    kinds = list(bg["models"].keys())
    kind = st.selectbox("Network used for diagnosis", kinds,
                        index=kinds.index("MLP-BP") if "MLP-BP" in kinds else 0)
    mode = st.radio("Motor under test", ["🎲 Random unknown motor", "Choose the condition myself"], horizontal=True)
    if mode.startswith("Choose"):
        c1, c2, c3 = st.columns(3)
        cond = c1.selectbox("Condition", range(len(CLASSES)), format_func=lambda i: CLASSES[i])
        sev = c2.select_slider("Severity", [1, 2, 3], 2, disabled=cond == 0)
        load = c3.slider("Load %", 20, 120, 75, 5) / 100
    noise = st.slider("Sensor noise (% of RMS)", 0.0, 3.0, 1.0, 0.1) / 100

    if st.button("🔍 Acquire signals & diagnose", type="primary"):
        rng = np.random.default_rng()
        if mode.startswith("🎲"):
            cond = int(rng.integers(len(CLASSES)))
            sev = int(rng.integers(1, 4))
            load = float(rng.uniform(0.3, 1.1))
        with st.spinner("Simulating the motor and extracting features..."):
            sig, mp = run_sim(cond, sev, load, noise, 3.0, False, int(rng.integers(1 << 30)))
            x = extract_features(sig)
        st.session_state.demo = (sig, x, cond, sev, load)

    if "demo" in st.session_state:
        sig, x, cond, sev, load = st.session_state.demo
        xg = bg["scaler"].transform(x[None, :])
        mg = bg["models"][kind]
        pg = mg.predict_proba(xg)[0]
        cg = int(np.argmax(pg))
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Detection")
            if bd is not None:
                md_ = bd["models"].get(kind) or list(bd["models"].values())[0]
                pdx = md_.predict_proba(bd["scaler"].transform(x[None, :]))[0]
                faulty = pdx[1] > 0.5
                (st.error if faulty else st.success)(
                    f"### {'FAULTY ⚠️' if faulty else 'HEALTHY ✅'}\nconfidence {max(pdx)*100:.1f} %")
            else:
                st.info("No detection network saved.")
            st.subheader("Diagnosis")
            msg = f"### {CLASSES[cg]} {ICONS[cg]}\nconfidence {pg[cg]*100:.1f} %"
            (st.success if cg == 0 else st.warning)(msg)
        with c2:
            show(plots.confidence_bars(pg, CLASSES, f"{mg.name} – output"))
        with st.expander("🔓 Reveal the true condition"):
            truth = CLASSES[cond] + ("" if cond == 0 else f" – {SEVERITY_LABELS[cond][sev]}")
            ok = "✔ correct" if cg == cond else "✘ wrong"
            st.write(f"**True:** {truth}, load {load*100:.0f} %  →  diagnosis {ok}")
        vt5 = (f"ANN VERDICT: FAULTY – {CLASSES[cg]}  ({pg[cg]*100:.0f} %)" if cg else
               f"ANN VERDICT: HEALTHY  ({pg[cg]*100:.0f} %)")
        show(plots.add_verdict(plots.current_spectrum(sig, None, x[FEATURE_NAMES.index('slip')],
                                                      title="Measured current spectrum"), vt5, cg != 0))
        st.dataframe(pd.DataFrame([x], columns=FEATURE_NAMES).round(4), width="stretch")
