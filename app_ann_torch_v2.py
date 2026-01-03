# app_ann_torch_v2.py
# Streamlit app for RC frame / pushover prediction using a PyTorch ensemble
# IMPORTANT: Training used log1p(clip(x,0,∞)) for both X and Y when enabled.
# So this app applies log1p preprocessing internally and uses expm1 for inverse.

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

import torch
import torch.nn as nn


# ----------------------------
# Page config (ONLY ONCE)
# ----------------------------
st.set_page_config(
    page_title="Pushover Predictor (ANN)",
    page_icon="🧱",
    layout="wide",
)

# Use the script directory so paths work in Streamlit Cloud
ART_DIR = Path(__file__).resolve().parent


# ----------------------------
# Utilities
# ----------------------------
def load_meta(path: Path) -> dict:
    try:
        m = joblib.load(path)
        if isinstance(m, dict):
            return m
        return {"_raw": m}
    except Exception as e:
        return {"_error": str(e)}


def pick_first_key(d: dict, keys: list):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def resolve_schema_ui(meta: dict):
    """
    Only shown if meta.joblib does NOT contain FEATURES/YVARS.
    No log options here (designers should never see/choose log).
    """
    st.warning("Feature/target names were not found in meta.joblib. Please provide them below (one-time setup).")
    st.write("Meta keys found:", list(meta.keys()))

    uploaded_csv = st.file_uploader(
        "Optionally upload a sample CSV of INPUT FEATURES (columns will be used):",
        type=["csv"],
    )

    features = []
    if uploaded_csv is not None:
        try:
            T = pd.read_csv(uploaded_csv, nrows=3)
            features = list(T.columns)
            st.success(f"Detected {len(features)} feature columns from CSV.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    feat_text = st.text_area("Feature names (comma-separated)", value=",".join(features))
    y_text = st.text_input("Target names (comma-separated)", value="")

    save_btn = st.button("Save schema into meta.joblib")

    FEATURES = [c.strip() for c in feat_text.split(",") if c.strip()]
    YVARS = [c.strip() for c in y_text.split(",") if c.strip()]

    if save_btn:
        if not FEATURES or not YVARS:
            st.error("Please provide both FEATURE names and TARGET names.")
        else:
            meta["X_columns"] = FEATURES
            meta["Y_columns"] = YVARS

            # Model trained with log transforms (fixed; not user-configurable)
            meta["log_transform_X"] = True
            meta["log_transform_Y"] = True

            joblib.dump(meta, ART_DIR / "meta.joblib")
            st.success("Saved! Please rerun the app (Rerun button or refresh browser).")

    # Return fixed True/True for log flags (trained that way)
    return FEATURES, YVARS, True, True


# ----------------------------
# Load artifacts (cache to speed reruns)
# ----------------------------
@st.cache_resource
def load_artifacts():
    meta = load_meta(ART_DIR / "meta.joblib")

    try:
        Xsc = joblib.load(ART_DIR / "Xsc.pkl")
    except Exception:
        Xsc = None

    try:
        Ysc = joblib.load(ART_DIR / "Ysc.pkl")
    except Exception:
        Ysc = None

    try:
        params = joblib.load(ART_DIR / "best_params.joblib")
        if not isinstance(params, dict):
            params = {}
    except Exception:
        params = {}

    try:
        fold_states = joblib.load(ART_DIR / "fold_states.joblib")
        if fold_states is None:
            fold_states = []
    except Exception:
        fold_states = []

    return meta, Xsc, Ysc, params, fold_states


meta, Xsc, Ysc, params, fold_states = load_artifacts()


# ----------------------------
# Resolve schema (feature/target names)
# ----------------------------
CAND_X = ["X_columns", "features", "feature_names", "X_cols", "X_df_columns", "input_columns"]
CAND_Y = ["Y_columns", "targets", "target_names", "Y_cols", "output_columns", "YVARS"]

FEATURES = pick_first_key(meta, CAND_X)
YVARS = pick_first_key(meta, CAND_Y)

# FIXED: trained with log transforms (and training uses log1p/clip)
log_X = True
log_Y = True


# ----------------------------
# Header
# ----------------------------
st.title("Pushover Curve Predictor (ANN)")
st.caption("Enter parameters in the shown units. Preprocessing (normalization/log) is handled automatically.")


# ----------------------------
# Guardrails
# ----------------------------
if FEATURES is None or YVARS is None:
    FEATURES, YVARS, log_X, log_Y = resolve_schema_ui(meta)
    st.stop()

if Xsc is None or Ysc is None:
    st.error("Missing scalers. Ensure Xsc.pkl and Ysc.pkl exist in the app folder.")
    st.stop()

if not isinstance(FEATURES, (list, tuple)) or not isinstance(YVARS, (list, tuple)):
    st.error("Schema is not a list/tuple. Ensure meta.joblib has X_columns and Y_columns as lists.")
    st.stop()

if not fold_states:
    st.error("Missing fold_states.joblib or it contains no model states.")
    st.stop()


# ----------------------------
# Model definitions
# ----------------------------
DEVICE = "cpu"

ACTS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


class MLP(nn.Module):
    def __init__(self, in_d, out_d, hidden, act="relu", drop=0.15):
        super().__init__()
        A = ACTS.get(act, nn.ReLU)
        layers = []
        p = in_d
        for h in hidden:
            layers += [nn.Linear(p, h), A(), nn.Dropout(drop)]
            p = h
        layers += [nn.Linear(p, out_d)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiHead(nn.Module):
    def __init__(self, in_d, out_d, trunk, head, act="relu", drop=0.15):
        super().__init__()
        A = ACTS.get(act, nn.ReLU)

        trunk_layers = []
        p = in_d
        for h in trunk:
            trunk_layers += [nn.Linear(p, h), A(), nn.Dropout(drop)]
            p = h
        self.trunk = nn.Sequential(*trunk_layers)

        self.heads = nn.ModuleList()
        for _ in range(out_d):
            head_layers = []
            ph = p
            for h in head:
                head_layers += [nn.Linear(ph, h), A()]
                ph = h
            head_layers += [nn.Linear(ph, 1)]
            self.heads.append(nn.Sequential(*head_layers))

    def forward(self, x):
        z = self.trunk(x)
        return torch.cat([h(z) for h in self.heads], dim=1)


def build_model(n_in, n_out, cfgp, use_multihead=True, head_sizes=None):
    hidden_sizes = cfgp.get("hidden_sizes", [50, 25, 10])
    activation = cfgp.get("activation", "elu")
    dropout = cfgp.get("dropout", 0.30)

    if use_multihead and n_out > 1:
        head_sizes = head_sizes or cfgp.get("head_sizes", [64]) or [64]
        model = MultiHead(n_in, n_out, hidden_sizes, head_sizes, activation, dropout)
    else:
        model = MLP(n_in, n_out, hidden_sizes, activation, dropout)

    return model.to(DEVICE)


def predict_with_states(X_np, fold_states, params, n_out):
    X_t = torch.from_numpy(X_np.astype(np.float32)).to(DEVICE)
    preds = []

    for sd in fold_states:
        mdl = build_model(
            n_in=X_np.shape[1],
            n_out=n_out,
            cfgp=params,
            use_multihead=params.get("use_multihead", True),
            head_sizes=params.get("head_sizes", [64]),
        )
        mdl.load_state_dict(sd)
        mdl.eval()
        with torch.no_grad():
            yp = mdl(X_t).cpu().numpy()
        preds.append(yp)

    return np.mean(preds, axis=0)


# ----------------------------
# Preprocessing (INTERNAL ONLY) — MATCH TRAINING EXACTLY
# Training code:
#   X = log1p(clip(X,0,∞))
#   Y = log1p(clip(Y,0,∞))
# Then scaling is applied on those transformed values.
# ----------------------------
def fwd_X(df_or_np):
    """Forward-transform raw inputs -> model input space."""
    X = df_or_np.values if hasattr(df_or_np, "values") else np.asarray(df_or_np)
    X = X.astype(np.float32, copy=False)

    if log_X:
        X = np.log1p(np.clip(X, a_min=0.0, a_max=None)).astype(np.float32)

    Xz = Xsc.transform(X)
    return Xz


def inv_Y(Yz):
    """Inverse-transform model outputs -> final engineering outputs."""
    Y = Ysc.inverse_transform(Yz)

    if log_Y:
        # inverse of log1p is expm1
        Y = np.expm1(Y).astype(np.float32)

    return Y


# ----------------------------
# UI labels + units
# ----------------------------
FEATURE_UI = {
    "NS": {"label": "Number of stories", "unit": "stories"},
    "BW": {"label": "Bay width", "unit": "mm"},
    "BN": {"label": "Number of bays", "unit": "count"},
    "FM": {"label": "Infill strength", "unit": "MPa"},
    "TM": {"label": "Infill thickness", "unit": "mm"},
    "IP": {"label": "Infill percentage", "unit": "%"},
    "IP_GS": {"label": "Infill percentage at ground storey", "unit": "%"},
    "FCK": {"label": "Concrete strength (fck)", "unit": "MPa"},
    "AC": {"label": "Area of column", "unit": "mm^2"},
    "AB": {"label": "Area of beam", "unit": "mm^2"},
    "rhoC": {"label": "Longitudinal reinforcement ratio (column)", "unit": "-"},
    "rhoB": {"label": "Longitudinal reinforcement ratio (beam)", "unit": "-"},
}


def nice_label(key: str) -> str:
    ui = FEATURE_UI.get(key, {})
    label = ui.get("label", key)
    unit = (ui.get("unit", "") or "").strip()
    if unit:
        return f"{label} ({key}) [{unit}]"
    return f"{label} ({key})"


# ----------------------------
# Sidebar info (designer-friendly)
# ----------------------------
st.sidebar.header("Model info")
st.sidebar.write(f"Inputs: {len(FEATURES)}")
st.sidebar.write(f"Outputs: {len(YVARS)}")
st.sidebar.write("Preprocessing: handled automatically")
st.sidebar.write(f"Hidden layers: {params.get('hidden_sizes', [50, 25, 10])}")
st.sidebar.write(f"Head sizes: {params.get('head_sizes', [64])}")


# ----------------------------
# Single prediction UI
# ----------------------------
st.subheader("Geometric and material properties of frame")
#st.info("Tip: Enter realistic design values. Very small/zero inputs can produce unreliable predictions.", icon="ℹ️")

cols = st.columns(3)
inputs = []

for i, name in enumerate(FEATURES):
    with cols[i % 3]:
        label = nice_label(name)

        # Integer counts must be >= 1
        if name in ["NS", "BN"]:
            val = st.number_input(label, min_value=1, value=1, step=1)

        # Non-negative for other variables (training clipped negatives to 0)
        else:
            val = st.number_input(label, min_value=0.0, value=0.0, format="%.6f")

        inputs.append(val)

if st.button("Predict (single)"):
    try:
        X_in = np.array(inputs, dtype=np.float32).reshape(1, -1)

        # Inform user if they entered negatives (shouldn't happen with min_value, but keep safe)
        if np.any(X_in < 0):
            st.warning("Some inputs were negative; they will be clipped to 0 internally (as in training).")

        Xz = fwd_X(X_in)

        # Optional sanity check: if scaled inputs are far outside training distribution
        zmax = float(np.max(np.abs(Xz)))
        if zmax > 6:
            st.warning(
                f"Inputs appear far from the training distribution (max |z| ≈ {zmax:.2f}). "
                "Please verify units and ranges.",
                icon="⚠️",
            )

        Yz = predict_with_states(Xz, fold_states, params, n_out=len(YVARS))
        Yo = inv_Y(Yz)

        df = pd.DataFrame(Yo, columns=YVARS)

        st.success("Prediction")
        st.dataframe(df.style.format("{:.6f}"))
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "ann_pred_single.csv",
            "text/csv",
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")


st.markdown("---")


# ----------------------------
# Batch prediction UI
# ----------------------------
st.subheader("Batch prediction (CSV)")
up = st.file_uploader(
    "Upload CSV with EXACT columns (same names and order as training features).",
    type=["csv"],
)

if up is not None:
    try:
        df_in = pd.read_csv(up)

        if list(df_in.columns) != list(FEATURES):
            st.error("CSV columns must match the training feature names AND order exactly.")
            st.write("Expected:", FEATURES)
            st.write("Found:", list(df_in.columns))
        else:
            Xz = fwd_X(df_in)

            zmax = float(np.max(np.abs(Xz)))
            if zmax > 6:
                st.warning(
                    f"Some rows appear far from the training distribution (max |z| ≈ {zmax:.2f}). "
                    "Please verify units and ranges.",
                    icon="⚠️",
                )

            Yz = predict_with_states(Xz, fold_states, params, n_out=len(YVARS))
            Yo = inv_Y(Yz)
            out = pd.DataFrame(Yo, columns=YVARS)

            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out.head().style.format("{:.6f}"))

            st.download_button(
                "Download predictions",
                out.to_csv(index=False),
                file_name="ann_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Failed to score file: {e}")
