"""Entrena en todo train_sup con la config robusta y escribe submit.csv."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "kit_zip/participant_kit")
from scoring import validate_prediction_package, FAULT_IDS, LABEL_COLUMNS
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier

RS = 42
D = "datos_zip/datos_sinraw/"
train = pd.read_parquet(D + "train_sup.parquet")
test = pd.read_parquet(D + "test.parquet")
FA = FAULT_IDS
lab = train[LABEL_COLUMNS].to_numpy()
estado = np.where(lab.sum(1) == 0, 0, lab.argmax(1) + 1)

BASE = [c for c in test.columns
        if c not in ("window_id", "machine_id", "session_id", "timestamp_start_s")]
ACC = ["acc_radial_a", "acc_radial_b", "acc_axial"]

def fisicas(df):
    X = pd.DataFrame(index=df.index)
    rpm = df.rpm_mean.clip(lower=1) / 1000.0
    rpm2 = rpm ** 2
    for a in ACC:
        rms = df[f"{a}__rms"].clip(lower=1e-9)
        X[f"{a}__rms_n"] = rms / rpm2
        X[f"{a}__1x_r"] = df[f"{a}__1x"] / (rms ** 2 + 1e-9)
        X[f"{a}__2x1x"] = df[f"{a}__2x"] / (df[f"{a}__1x"].abs() + 1e-6)
        X[f"{a}__lkurt"] = np.log1p(df[f"{a}__kurtosis"].clip(lower=0))
        X[f"{a}__crest_n"] = df[f"{a}__crest"]
    ra = df["acc_radial_a__rms"].clip(lower=1e-9)
    X["ax_rad"] = df["acc_axial__rms"] / ra
    X["rad_b_a"] = df["acc_radial_b__rms"] / ra
    X["vib_tot"] = df[[f"{a}__rms" for a in ACC]].sum(1) / rpm2
    V = df[["voltage_a__rms", "voltage_b__rms", "voltage_c__rms"]]
    I = df[["current_a__rms", "current_b__rms", "current_c__rms"]]
    vm, im = V.mean(1).clip(lower=1e-9), I.mean(1).clip(lower=1e-9)
    X["V_desbal"] = (V.max(1) - V.min(1)) / vm
    X["I_desbal"] = (I.max(1) - I.min(1)) / im
    X["V_cv"] = V.std(1) / vm
    X["I_cv"] = I.std(1) / im
    X["I_min_r"] = I.min(1) / im
    X["I_max_r"] = I.max(1) / im
    X["S_ap"] = im * vm / 1000.0
    X["I_por_rpm"] = im / rpm
    X["I_por_flow"] = im / (df.flow_mean.abs() + 1e-6)
    X["dT_wind"] = df["temp_winding__mean"] - df.Tamb
    X["dT_b1"] = df["temp_bearing1__mean"] - df.Tamb
    X["dT_b2"] = df["temp_bearing2__mean"] - df.Tamb
    X["dT_b1b2"] = df["temp_bearing1__mean"] - df["temp_bearing2__mean"]
    X["dT_wind_b"] = df["temp_winding__mean"] - df[["temp_bearing1__mean", "temp_bearing2__mean"]].mean(1)
    X["dT_wind_rpm"] = X["dT_wind"] / rpm
    X["head_n"] = df.delta_p_mean / rpm2
    X["flow_n"] = df.flow_mean / rpm
    X["p_in_n"] = df.pressure_in_mean / rpm2
    X["p_ratio"] = df.pressure_out_mean / (df.pressure_in_mean.abs() + 1e-6)
    X["hidr_pot"] = df.flow_mean * df.delta_p_mean / (im * vm + 1e-6)
    X["flow_head"] = df.flow_mean / (df.delta_p_mean.abs() + 1e-6)
    X["rpm_cv"] = df.rpm_std / df.rpm_mean.clip(lower=1)
    nanc = [c for c in df.columns if c.endswith("__nan_frac")]
    X["nan_tot"] = df[nanc].sum(1)
    X["nan_max"] = df[nanc].max(1)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def zmaq(df, cols):
    """Desviacion de cada ventana respecto de su propia maquina. Intra-maquina por
    construccion: nunca mezcla train con test, y no mira ninguna etiqueta."""
    g = df.groupby("machine_id")[cols]
    z = (df[cols] - g.transform("median")) / (g.transform("std") + 1e-9)
    z.columns = [c + "__z" for c in cols]
    return z

def construir(df):
    fis = pd.concat([df[BASE], fisicas(df)], axis=1)
    tmp = fis.copy(); tmp["machine_id"] = df["machine_id"].to_numpy()
    return pd.concat([fis, zmaq(tmp, list(fis.columns))], axis=1)

Xtr, Xte = construir(train), construir(test)
assert list(Xtr.columns) == list(Xte.columns), "columnas desalineadas"
print("features:", Xtr.shape[1], "| train", Xtr.shape, "| test", Xte.shape)

MODELOS = [
    lambda: lgb.LGBMClassifier(objective="multiclass", num_class=14, n_estimators=500,
        learning_rate=0.05, num_leaves=15, min_child_samples=25, subsample=0.9,
        subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, random_state=RS, deterministic=True,
        force_col_wise=True, n_jobs=1),
    lambda: ExtraTreesClassifier(n_estimators=600, min_samples_leaf=3,
        max_features="sqrt", n_jobs=1, random_state=RS),
]
P = np.zeros((len(test), 14))
for ctor in MODELOS:
    m = ctor(); m.fit(Xtr, estado); P += m.predict_proba(Xte)
P /= len(MODELOS)

# promedio geometrico por sesion: la etiqueta es constante en las 7 ventanas
L = pd.DataFrame(np.log(np.clip(P, 1e-9, 1)))
L["s"] = test.session_id.to_numpy()
P = np.exp(L.groupby("s").transform("mean").to_numpy())
P /= P.sum(1, keepdims=True)

sub = pd.DataFrame({"window_id": test.window_id.to_numpy()})
for j, f in enumerate(FA):
    sub[f] = np.clip(P[:, j + 1], 0, 1)
sub = validate_prediction_package(test, sub)
sub.to_csv("submit.csv", index=False)
print("submit.csv:", sub.shape, "-> validado")
print("suma de probas por fila: media %.3f | prevalencia train %.3f"
      % (sub[FA].sum(1).mean(), train[LABEL_COLUMNS].mean().sum()))
print(sub[FA].mean().round(4).to_string())
