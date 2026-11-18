"""
LUMINA end-to-end pipeline for Colab (Stage 1 and Stage 2 both Keras).
  Stage 1 (v18): light curve -> Planet / Eclipsing Binary / Blend / Artifact
  Stage 2: ABC-style spectrum -> 6 atmospheric parameters (+ posterior std)
Stage 2 only runs if Stage 1 says Planet (confidently) AND a spectrum is supplied.

Colab usage (new notebook, e.g. LUMINA_integration):
    from google.colab import drive; drive.mount('/content/drive')
    !pip install -q lightkurve
    %run /content/drive/MyDrive/exovision_data/lumina_pipeline.py
    pipe = LuminaPipeline()
    pipe.run_from_pickle("<path to a features_kepler_robovetter pkl>", spectrum=demo_spectrum(0))
"""
import os
import pickle
import numpy as np
import keras
import tensorflow as tf

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
SAVE_DIR = "/content/drive/MyDrive/exovision_data"

# --- Stage 1 (from the v18 training script) ---
STAGE1_PATH = f"{SAVE_DIR}/exovision_model_v18_all_data_4class.keras"
SCALAR_NORM_PATH = f"{SAVE_DIR}/scalar_normalization_v18.pkl"      # {"mean","std","keys"}
CLASSES = ["Planet", "Eclipsing Binary", "Blend", "Artifact"]
LOCAL_BINS, GLOBAL_BINS = 201, 2001
SCALAR_KEYS = ["period", "epoch", "depth", "duration", "flux_std",
               "flux_mean", "flux_median", "flux_min", "flux_max", "model_snr"]
PLANET_CONF_THRESHOLD = 0.60

# --- Stage 2 (files saved by the spectroscopy notebook) ---
S2_DIR = f"{SAVE_DIR}/abc_features"
STAGE2_PATH = f"{S2_DIR}/exovision_abc_multitask.keras"
SCALERS_PATH = f"{S2_DIR}/abc_scalers.pkl"
FEATURES_NPZ = f"{S2_DIR}/abc_features.npz"
# Usual ABC target order; true_mean ranges (138..5375 and -10..-3) are consistent with it.
PARAM_NAMES = ["temperature_K", "log_H2O", "log_CO2", "log_CO", "log_CH4", "log_NH3"]


# ======================= custom Keras layers used by Stage 2 =================
@keras.saving.register_keras_serializable(package="Custom")
class Corrupt(keras.layers.Layer):
    """Training-time masking/noise layer (DAE only). Identity at inference."""
    def __init__(self, mask_rate=0.1, noise_std=0.1, **kw):
        super().__init__(**kw)
        self.mask_rate, self.noise_std = mask_rate, noise_std

    def call(self, x, training=None):
        return x

    def get_config(self):
        c = super().get_config()
        c.update(mask_rate=self.mask_rate, noise_std=self.noise_std)
        return c


@keras.saving.register_keras_serializable(package="Custom")
class PosEmb(keras.layers.Layer):
    """Learned positional embedding: one (1, L, D) weight added to the input.
    (Reconstructed by loading test; reproduces the saved R2 within 0.001.)"""
    def __init__(self, **kw):
        kw.pop("activity_regularizer", None)
        extra = {k: kw.pop(k) for k in list(kw) if k not in ("name", "trainable", "dtype")}
        super().__init__(**kw)
        self.extra = extra

    def build(self, input_shape):
        self.pos = self.add_weight(shape=(1, input_shape[1], input_shape[2]),
                                   initializer="zeros", name="pos")

    def call(self, x):
        return x + self.pos

    def get_config(self):
        c = super().get_config()
        c.update(self.extra)
        return c


# ============================ Stage 1 ========================================
def load_stage1(path=STAGE1_PATH, build_fn=None, width=None):
    """Load the v18 .keras model; fall back to rebuild + load_weights if needed."""
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"Direct load failed ({type(e).__name__}); trying rebuild + load_weights...")
        if build_fn is None or width is None:
            raise RuntimeError("Pass build_fn=build_model and width=<best_width> to rebuild.") from e
        model = build_fn(width=width)
        model.load_weights(path)
        return model


def featurize_light_curve(target, mission="Kepler", period=None, epoch=None, duration_days=None):
    """
    Fresh target -> local[201], global[2001], scalars[10].
    WARNING: approximation of the training featurizer (window lengths, binning, how
    flux_* scalars and model_snr were computed). Prefer run_from_pickle() or call your
    own featurizer and pass its output to run_features().
    """
    import lightkurve as lk
    sr = lk.search_lightcurve(target, mission=mission)
    if len(sr) == 0:
        raise ValueError(f"No light curves found for {target} ({mission})")
    lc = sr.download_all().stitch().remove_nans().remove_outliers(sigma=5)
    lc = lc.flatten(window_length=801, polyorder=2)

    if period is None or epoch is None:                       # BLS fallback
        bls = lc.to_periodogram(method="bls", period=np.linspace(0.5, 100, 20000))
        period = float(bls.period_at_max_power.value)
        epoch = float(bls.transit_time_at_max_power.value)
        duration_days = float(bls.duration_at_max_power.value)
    duration_days = duration_days or 0.1 * period

    folded = lc.fold(period=period, epoch_time=epoch)
    phase = folded.phase.value / period
    flux = folded.flux.value - np.nanmedian(folded.flux.value)      # baseline 0.0

    def bin_view(lo, hi, n):
        edges = np.linspace(lo, hi, n + 1)
        idx = np.clip(np.digitize(phase, edges) - 1, 0, n - 1)
        out = np.full(n, np.nan)
        for b in range(n):
            m = idx == b
            if m.any():
                out[b] = np.median(flux[m])
        ok = ~np.isnan(out)
        if ok.sum() < 2:
            return np.zeros(n, np.float32)
        out[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), out[ok])
        return out.astype(np.float32)

    half = 2.0 * duration_days / period
    local, glob = bin_view(-half, half, LOCAL_BINS), bin_view(-0.5, 0.5, GLOBAL_BINS)
    in_tr = np.abs(phase) < (duration_days / period) / 2
    depth = float(-np.nanmedian(flux[in_tr])) if in_tr.any() else 0.0
    raw = lc.flux.value
    snr = depth / (np.nanstd(flux) / np.sqrt(max(int(in_tr.sum()), 1)) + 1e-12)
    feats = {"period": period, "epoch": epoch, "depth": depth, "duration": duration_days,
             "flux_std": np.nanstd(raw), "flux_mean": np.nanmean(raw),
             "flux_median": np.nanmedian(raw), "flux_min": np.nanmin(raw),
             "flux_max": np.nanmax(raw), "model_snr": snr}
    return {"local": local, "global_": glob,
            "scalars": np.array([feats[k] for k in SCALAR_KEYS], np.float32)}


def feats_from_pickle(path):
    """Load a feature pickle made by the Stage 1 featurizer (same format as training data)."""
    with open(path, "rb") as fh:
        rec = pickle.load(fh)
    return {"local": np.asarray(rec["local_view"], np.float32),
            "global_": np.asarray(rec["global_view"], np.float32),
            "scalars": np.array([rec[k] for k in SCALAR_KEYS], np.float32),
            "true_label": rec.get("label")}


# ============================ Stage 2 ========================================
def build_spectrum_input(sc, wavelength, depth, noise, width=None):
    """
    Build the (1, 52, 5) Stage 2 input exactly as in training (verified: max diff 0.0 on
    every channel against abc_features.npz). Dataset-level per-bin standardization:
      ch0 = z(spec)          ch1 = z(log10(noise))   ch2 = z(spec / noise)
      ch3 = z(bin width)     ch4 = min-max(wavelength) on the training wavelength grid
    """
    wl = np.asarray(wavelength, np.float64).reshape(-1)
    spec = np.asarray(depth, np.float64).reshape(-1)
    noise = np.asarray(noise, np.float64).reshape(-1)
    if len(wl) != 52:
        raise ValueError(f"expected 52 wavelength bins, got {len(wl)}")
    if width is None:                                          # fallback if width not given
        width = np.gradient(wl)
    width = np.asarray(width, np.float64).reshape(-1)

    def z(key, arr):
        m, s = [np.asarray(v, np.float64).reshape(-1) for v in sc["inputs"][key]]
        return (arr - m) / s

    grid = np.asarray(sc["wl"], np.float64).reshape(-1)
    ch = [z("spec", spec),
          z("log_noise", np.log10(noise)),
          z("snr", spec / noise),
          z("width", width),
          (wl - grid.min()) / (grid.max() - grid.min())]
    return np.stack(ch, axis=-1).astype(np.float32)[None]      # (1, 52, 5)


def demo_spectrum(i=0, split="test"):
    """A real labeled ABC spectrum from the saved dataset (for demos / sanity checks)."""
    z = np.load(FEATURES_NPZ)
    idx = int(z[f"{split}_idx"][i])
    return {"wavelength": z["wl"], "depth": z["spec"][idx], "noise": z["noise"][idx],
            "width": z["width"][idx],
            "truth": {n: float(v) for n, v in zip(PARAM_NAMES, z["t_tracedata_mean"][idx])}}


def habitability_indicator(p):
    """Post-hoc heuristic from PREDICTED values only. Not a trained output."""
    T, score, notes = p.get("temperature_K", np.nan), 0.0, []
    if 200 <= T <= 400:
        score += 0.5; notes.append("temperature in liquid-water range")
    elif 150 <= T < 200 or 400 < T <= 600:
        score += 0.2; notes.append("temperature marginal")
    else:
        notes.append("temperature outside habitable range")
    if p.get("log_H2O", -99) > -6:
        score += 0.3; notes.append("water vapour present")
    if p.get("log_CH4", -99) > -7 and p.get("log_CO2", -99) > -7:
        score += 0.2; notes.append("CH4 + CO2 present")
    return {"score": round(min(score, 1.0), 2), "notes": notes}


class Stage2Predictor:
    """Spectrum -> 6 atmospheric parameters. The multitask model has its own encoder, so the
    DAE is not needed at inference (the multitask model alone reproduces the saved R2)."""

    def __init__(self, model_path=STAGE2_PATH, scalers_path=SCALERS_PATH):
        self.model = keras.models.load_model(
            model_path, compile=False, custom_objects={"Corrupt": Corrupt, "PosEmb": PosEmb})
        with open(scalers_path, "rb") as fh:
            self.sc = pickle.load(fh)
        t = lambda k: [np.asarray(v, np.float32).reshape(-1) for v in self.sc["targets"][k]]
        self.m_mean, self.m_std = t("tracedata_mean")          # (mean, std) of target means - verified
        self.s_mean, self.s_std = t("tracedata_std")           # same for posterior std - see note below

    def predict_x(self, X):                                    # X: (N, 52, 5)
        p = self.model.predict(X, batch_size=256, verbose=0)
        mean = p[:, :6] * self.m_std + self.m_mean
        std = np.maximum(p[:, 6:] * self.s_std + self.s_mean, 0.0)   # de-normalization UNVERIFIED
        return mean, std

    def predict_spectrum(self, spectrum):
        X = build_spectrum_input(self.sc, spectrum["wavelength"], spectrum["depth"],
                                 spectrum["noise"], spectrum.get("width"))
        mean, std = self.predict_x(X)
        return mean[0], std[0]


# ============================ Pipeline =======================================
class LuminaPipeline:
    def __init__(self, stage1_build_fn=None, stage1_width=None, load_stage2=True):
        self.s1 = load_stage1(build_fn=stage1_build_fn, width=stage1_width)
        with open(SCALAR_NORM_PATH, "rb") as fh:
            nrm = pickle.load(fh)
        assert list(nrm["keys"]) == SCALAR_KEYS, f"scalar key mismatch: {nrm['keys']}"
        self.s_mean, self.s_std = nrm["mean"].reshape(-1), nrm["std"].reshape(-1)
        self.s2 = Stage2Predictor() if load_stage2 else None

    def classify(self, feats):
        sc = ((feats["scalars"] - self.s_mean) / self.s_std).astype(np.float32)
        probs = self.s1.predict({"local_view": feats["local"][None, :, None],
                                 "global_view": feats["global_"][None, :, None],
                                 "scalar_features": sc[None]}, verbose=0)[0]
        k = int(np.argmax(probs))
        return {"label": CLASSES[k], "confidence": float(probs[k]),
                "probs": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)}}

    def characterize(self, spectrum):
        mean, std = self.s2.predict_spectrum(spectrum)
        params = {n: round(float(v), 3) for n, v in zip(PARAM_NAMES, mean)}
        res = {"params": params, "uncertainty": {n: round(float(v), 3) for n, v in zip(PARAM_NAMES, std)},
               "habitability": habitability_indicator(params)}
        if "truth" in spectrum:
            res["truth_for_demo_spectrum"] = {n: round(v, 3) for n, v in spectrum["truth"].items()}
        return res

    def run_features(self, feats, name="target", spectrum=None):
        s1 = self.classify(feats)
        out = {"target": name, "stage1": s1, "stage2": None}
        if feats.get("true_label"):
            out["true_label"] = feats["true_label"]
        if s1["label"] != "Planet":
            out["note"] = f"Classified as {s1['label']} - Stage 2 skipped."
        elif s1["confidence"] < PLANET_CONF_THRESHOLD:
            out["note"] = f"Planet confidence {s1['confidence']:.2f} below threshold - Stage 2 skipped."
        elif spectrum is None:
            out["note"] = "Planet confirmed but no spectrum supplied - Stage 2 skipped."
        elif self.s2 is None:
            out["note"] = "Stage 2 model not loaded."
        else:
            out["stage2"] = self.characterize(spectrum)
            out["note"] = "Spectrum is a supplied/simulated ABC spectrum, not measured for this target."
        return out

    def run_from_pickle(self, pkl_path, spectrum=None):
        return self.run_features(feats_from_pickle(pkl_path), os.path.basename(pkl_path)[:-4], spectrum)

    def run(self, target, mission="Kepler", spectrum=None, period=None, epoch=None, duration_days=None):
        feats = featurize_light_curve(target, mission, period, epoch, duration_days)
        return self.run_features(feats, target, spectrum)
