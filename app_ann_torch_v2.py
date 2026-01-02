import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import torch
import torch.nn as nn

ART_DIR = Path(".")

def load_meta(path):
    try:
        m = joblib.load(path)
        if isinstance(m, dict):
            return m
        else:
            return {"_raw": m}
    except Exception as e:
        return {"_error": str(e)}

def pick_first_key(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def resolve_schema_ui(meta):
    st.warning("Could not find feature/target names in meta.joblib. Please provide them below.")
    st.write("Meta keys found:", list(meta.keys()))
    uploaded_csv = st.file_uploader("Optionally upload a sample CSV of input features (columns will be used):", type=["csv"])

    features = []
    if uploaded_csv is not None:
        try:
            T = pd.read_csv(uploaded_csv, nrows=3)
            features = list(T.columns)
            st.success(f"Detected {len(features)} feature columns from CSV.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    feat_text = st.text_area("Feature names (comma-separated)", value=",".join(features))
    y_text    = st.text_input("Target names (comma-separated)", value="")

    col1, col2 = st.columns(2)
    with col1:
        log_x = st.checkbox("log_transform_X", value=bool(meta.get("log_transform_X", False)))
    with col2:
        log_y = st.checkbox("log_transform_Y", value=bool(meta.get("log_transform_Y", False)))

    save_btn = st.button("Save schema into meta.joblib")
    FEATURES = [c.strip() for c in feat_text.split(",") if c.strip()]
    YVARS    = [c.strip() for c in y_text.split(",") if c.strip()]
    if save_btn:
        if not FEATURES or not YVARS:
            st.error("Please provide both FEATURES and YVARS.")
        else:
            meta["X_columns"] = FEATURES
            meta["Y_columns"] = YVARS
            meta["log_transform_X"] = bool(log_x)
            meta["log_transform_Y"] = bool(log_y)
            joblib.dump(meta, ART_DIR / "meta.joblib")
            st.success("Saved! Please rerun the app (Ctrl+R in browser or 'Rerun' button).")
    return FEATURES, YVARS, bool(meta.get("log_transform_X", False)), bool(meta.get("log_transform_Y", False))

# Load artifacts
meta = load_meta(ART_DIR / "meta.joblib")
try:
    Xsc = joblib.load(ART_DIR / "Xsc.pkl")
    Ysc = joblib.load(ART_DIR / "Ysc.pkl")
except Exception as e:
    Xsc = None; Ysc = None

try:
    params = joblib.load(ART_DIR / "best_params.joblib")
except Exception as e:
    params = {}

try:
    fold_states = joblib.load(ART_DIR / "fold_states.joblib")
except Exception as e:
    fold_states = []

# Resolve schema
CAND_X = ["X_columns","features","feature_names","X_cols","X_df_columns","input_columns"]
CAND_Y = ["Y_columns","targets","target_names","Y_cols","output_columns","YVARS"]

FEATURES = pick_first_key(meta, CAND_X)
YVARS    = pick_first_key(meta, CAND_Y)
log_X    = bool(meta.get("log_transform_X", False) or meta.get("log_x", False))
log_Y    = bool(meta.get("log_transform_Y", False) or meta.get("log_y", False))

st.set_page_config(page_title="ANN Predictor (PyTorch)", layout="wide")
st.title("ANN Predictor — PyTorch (Ensemble of CV folds)")

if FEATURES is None or YVARS is None:
    FEATURES, YVARS, log_X, log_Y = resolve_schema_ui(meta)
    st.stop()

if Xsc is None or Ysc is None or not isinstance(FEATURES, (list, tuple)) or not isinstance(YVARS, (list, tuple)):
    st.error("Missing scalers or schema. Ensure Xsc.pkl, Ysc.pkl and meta.joblib with FEATURES/YVARS are present.")
    st.stop()

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
        A = ACTS[act]
        L = []
        p = in_d
        for h in hidden:
            L += [nn.Linear(p, h), A()]
            L += [nn.Dropout(drop)]
            p = h
        L += [nn.Linear(p, out_d)]
        self.net = nn.Sequential(*L)
    def forward(self, x):
        return self.net(x)

class MultiHead(nn.Module):
    def __init__(self, in_d, out_d, trunk, head, act="relu", drop=0.15):
        super().__init__()
        A = ACTS[act]
        T = []
        p = in_d
        for h in trunk:
            T += [nn.Linear(p, h), A()]
            T += [nn.Dropout(drop)]
            p = h
        self.trunk = nn.Sequential(*T)
        self.heads = nn.ModuleList()
        for _ in range(out_d):
            H = []
            ph = p
            for h in head:
                H += [nn.Linear(ph, h), A()]
                ph = h
            H += [nn.Linear(ph, 1)]
            self.heads.append(nn.Sequential(*H))
    def forward(self, x):
        z = self.trunk(x)
        return torch.cat([h(z) for h in self.heads], dim=1)

def build_model(n_in, n_out, cfgp, use_multihead=True, head_sizes=None):
    hidden_sizes = cfgp.get("hidden_sizes", [50, 25, 10])
    activation   = cfgp.get("activation", "elu")
    dropout      = cfgp.get("dropout", 0.30)
    if use_multihead and n_out > 1:
        head_sizes = head_sizes or [64]
        model = MultiHead(n_in, n_out, hidden_sizes, head_sizes, activation, dropout)
    else:
        model = MLP(n_in, n_out, hidden_sizes, activation, dropout)
    return model.to(DEVICE)

def predict_with_states(X_np, fold_states, params, n_out):
    X_t = torch.from_numpy(X_np.astype(np.float32)).to(DEVICE)
    preds = []
    for sd in fold_states:
        mdl = build_model(
            n_in=X_np.shape[1], n_out=n_out, cfgp=params,
            use_multihead=params.get("use_multihead", True),
            head_sizes=params.get("head_sizes", [64])
        )
        mdl.load_state_dict(sd)
        mdl.eval()
        with torch.no_grad():
            yp = mdl(X_t).cpu().numpy()
        preds.append(yp)
    return np.mean(preds, axis=0)

def fwd_X(df_or_np):
    X = df_or_np.values if hasattr(df_or_np, "values") else np.asarray(df_or_np)
    X = X.astype(np.float32, copy=False)
    if log_X:
        eps = 1e-9
        X = np.log(X + eps)
    Xz = Xsc.transform(X)
    return Xz

def inv_Y(Yz):
    Y = Ysc.inverse_transform(Yz)
    if log_Y:
        Y = np.exp(Y)
    return Y

st.sidebar.header("Model info")
st.sidebar.write(f"Inputs: {len(FEATURES)}")
st.sidebar.write(f"Outputs: {len(YVARS)}")
st.sidebar.write(f"Log X: {log_X} | Log Y: {log_Y}")
st.sidebar.write(f"Hidden: {params.get('hidden_sizes', [50,25,10])}")
st.sidebar.write(f"Head sizes: {params.get('head_sizes', [64])}")

st.subheader("Single prediction")
cols = st.columns(3)
inputs = []
for i, name in enumerate(FEATURES):
    with cols[i % 3]:
        val = st.number_input(name, value=0.0, format="%.6f")
        inputs.append(val)

if st.button("Predict (single)"):
    X_in = np.array(inputs, dtype=np.float32).reshape(1, -1)
    Xz   = fwd_X(X_in)
    Yz   = predict_with_states(Xz, fold_states, params, n_out=len(YVARS))
    Yo   = inv_Y(Yz)
    df   = pd.DataFrame(Yo, columns=YVARS)
    st.success("Prediction")
    st.dataframe(df.style.format("{:.6f}"))
    st.download_button("Download CSV", df.to_csv(index=False), "ann_pred_single.csv", "text/csv")

st.markdown("---")

st.subheader("Batch prediction (CSV)")
up = st.file_uploader("Upload CSV with EXACT columns (same order) as training features", type=["csv"])
if up is not None:
    try:
        df_in = pd.read_csv(up)
        if list(df_in.columns) != list(FEATURES):
            st.error("CSV columns must match the training feature names and order.")
        else:
            Xz = fwd_X(df_in)
            Yz = predict_with_states(Xz, fold_states, params, n_out=len(YVARS))
            Yo = inv_Y(Yz)
            out = pd.DataFrame(Yo, columns=YVARS)
            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out.head().style.format("{:.6f}"))
            st.download_button("Download predictions", out.to_csv(index=False),
                               file_name="ann_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Failed to score file: {e}")
