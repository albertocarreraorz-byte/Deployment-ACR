from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "DP L12 2025 & 2026.xlsx"
MODEL_OUT = ROOT / "modelo_xgboost_paros_l12.json"
METADATA_OUT = ROOT / "modelo_xgboost_paros_l12_metadata.json"
FORECAST_OUT = ROOT / "pronostico_paros_l12_30_dias.csv"
REPORT_OUT = ROOT / "modelo_prediccion_paros_l12.html"

TARGET = "horas_paro"
FORECAST_DAYS = 30
LAGS = [1, 2, 3, 7, 14, 21, 28]
ROLLS = [7, 14, 28]


def esc(value) -> str:
    return html.escape(str(value))


def fmt(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.{decimals}f}"


def load_daily_data() -> pd.DataFrame:
    raw = pd.read_excel(SOURCE, sheet_name="Sheet1")
    raw["Fecha contable"] = pd.to_datetime(raw["Fecha contable"], errors="coerce")
    raw = raw.dropna(subset=["Fecha contable"]).copy()
    raw["duracion_min"] = pd.to_numeric(raw["Tiempo formal"], errors="coerce").fillna(
        pd.to_numeric(raw["Minutos de paro"], errors="coerce")
    )
    raw["duracion_min"] = raw["duracion_min"].fillna(0).clip(lower=0)
    raw["Tipo de paro"] = raw["Tipo de paro"].fillna("Sin clasificar")

    daily = (
        raw.groupby(raw["Fecha contable"].dt.normalize())
        .agg(
            horas_paro=("duracion_min", lambda s: s.sum() / 60),
            eventos=("duracion_min", "size"),
            mttr_min=("duracion_min", "mean"),
            tipo_dominante=("Tipo de paro", lambda s: s.value_counts().index[0]),
        )
        .rename_axis("fecha")
        .reset_index()
    )

    full_dates = pd.DataFrame(
        {"fecha": pd.date_range(daily["fecha"].min(), daily["fecha"].max(), freq="D")}
    )
    daily = full_dates.merge(daily, on="fecha", how="left")
    daily["horas_paro"] = daily["horas_paro"].fillna(0)
    daily["eventos"] = daily["eventos"].fillna(0)
    daily["mttr_min"] = daily["mttr_min"].fillna(0)
    daily["tipo_dominante"] = daily["tipo_dominante"].fillna("Sin paro")
    return daily


def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.copy()
    data["dia_semana"] = data["fecha"].dt.dayofweek
    data["mes"] = data["fecha"].dt.month
    data["dia_mes"] = data["fecha"].dt.day
    data["semana_anio"] = data["fecha"].dt.isocalendar().week.astype(int)
    data["trimestre"] = data["fecha"].dt.quarter
    data["es_fin_semana"] = (data["dia_semana"] >= 5).astype(int)
    data["dias_desde_inicio"] = (data["fecha"] - data["fecha"].min()).dt.days
    data["sin_dia"] = np.sin(2 * np.pi * data["dia_semana"] / 7)
    data["cos_dia"] = np.cos(2 * np.pi * data["dia_semana"] / 7)
    data["sin_mes"] = np.sin(2 * np.pi * data["mes"] / 12)
    data["cos_mes"] = np.cos(2 * np.pi * data["mes"] / 12)

    for lag in LAGS:
        data[f"lag_{lag}d"] = data[TARGET].shift(lag)
        data[f"eventos_lag_{lag}d"] = data["eventos"].shift(lag)

    for window in ROLLS:
        shifted = data[TARGET].shift(1)
        data[f"media_{window}d"] = shifted.rolling(window, min_periods=1).mean()
        data[f"max_{window}d"] = shifted.rolling(window, min_periods=1).max()
        data[f"eventos_media_{window}d"] = data["eventos"].shift(1).rolling(window, min_periods=1).mean()

    data["paro_acumulado_7d"] = data[TARGET].shift(1).rolling(7, min_periods=1).sum()
    data["paro_acumulado_28d"] = data[TARGET].shift(1).rolling(28, min_periods=1).sum()

    feature_cols = [c for c in data.columns if c not in {"fecha", TARGET, "tipo_dominante"}]
    return data.dropna(subset=feature_cols).reset_index(drop=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": float(r2)}


def build_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    ignore = {"fecha", TARGET, "tipo_dominante", "eventos", "mttr_min"}
    features = [c for c in data.columns if c not in ignore]
    return data[features], data[TARGET], features


def forecast(model: xgb.Booster, history: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    work = history.copy()
    preds = []

    for _ in range(FORECAST_DAYS):
        next_date = work["fecha"].max() + pd.Timedelta(days=1)
        new_row = pd.DataFrame(
            {
                "fecha": [next_date],
                "horas_paro": [np.nan],
                "eventos": [0],
                "mttr_min": [0],
                "tipo_dominante": ["Pronostico"],
            }
        )
        candidate = pd.concat([work, new_row], ignore_index=True)
        featured = add_features(candidate)
        row = featured[featured["fecha"] == next_date]
        pred = float(model.predict(xgb.DMatrix(row[features], feature_names=features))[0])
        pred = max(pred, 0)
        preds.append({"fecha": next_date, "horas_paro_pred": pred})
        work.loc[len(work)] = {
            "fecha": next_date,
            "horas_paro": pred,
            "eventos": 0,
            "mttr_min": 0,
            "tipo_dominante": "Pronostico",
        }

    result = pd.DataFrame(preds)
    result["fecha"] = result["fecha"].dt.date
    return result


def line_svg(actual: pd.DataFrame, validation: pd.DataFrame, forecast_df: pd.DataFrame) -> str:
    recent = actual.tail(120)[["fecha", TARGET]].rename(columns={TARGET: "valor"})
    val = validation[["fecha", "pred"]].rename(columns={"pred": "valor"})
    fc = forecast_df.copy()
    fc["fecha"] = pd.to_datetime(fc["fecha"])
    fc = fc.rename(columns={"horas_paro_pred": "valor"})

    width, height = 980, 330
    pad_l, pad_r, pad_t, pad_b = 56, 24, 30, 44
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    start = recent["fecha"].min()
    end = fc["fecha"].max()
    span = max((end - start).days, 1)
    ymax = max(float(recent["valor"].max()), float(val["valor"].max()), float(fc["valor"].max()), 1)

    def xy(date, value):
        x_pos = pad_l + ((date - start).days / span) * plot_w
        y_pos = height - pad_b - (float(value) / ymax) * plot_h
        return x_pos, y_pos

    def path(data: pd.DataFrame) -> str:
        commands = []
        for idx, row in data.iterrows():
            x_pos, y_pos = xy(pd.to_datetime(row["fecha"]), row["valor"])
            commands.append(("M" if len(commands) == 0 else "L") + f"{x_pos:.1f},{y_pos:.1f}")
        return " ".join(commands)

    grid = []
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y_pos = height - pad_b - frac * plot_h
        grid.append(f"<line x1='{pad_l}' y1='{y_pos:.1f}' x2='{width-pad_r}' y2='{y_pos:.1f}' stroke='#e2e8f0'/>")
        grid.append(f"<text x='{pad_l-10}' y='{y_pos+4:.1f}' text-anchor='end'>{fmt(ymax*frac,0)}</text>")

    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Pronostico de horas de paro'>"
        + "".join(grid)
        + f"<line x1='{pad_l}' y1='{height-pad_b}' x2='{width-pad_r}' y2='{height-pad_b}' stroke='#cbd5e1'/>"
        + f"<path d='{path(recent)}' fill='none' stroke='#2563eb' stroke-width='2.4'/>"
        + f"<path d='{path(val)}' fill='none' stroke='#f97316' stroke-width='2.4' stroke-dasharray='6 4'/>"
        + f"<path d='{path(fc)}' fill='none' stroke='#0f766e' stroke-width='2.8'/>"
        + "<rect x='645' y='14' width='12' height='12' fill='#2563eb'/><text x='664' y='25'>Real reciente</text>"
        + "<rect x='755' y='14' width='12' height='12' fill='#f97316'/><text x='774' y='25'>Validacion</text>"
        + "<rect x='855' y='14' width='12' height='12' fill='#0f766e'/><text x='874' y='25'>Pronostico</text>"
        + f"<text x='{pad_l}' y='{height-14}'>{start:%d/%m/%Y}</text><text x='{width-pad_r}' y='{height-14}' text-anchor='end'>{end:%d/%m/%Y}</text>"
        + "</svg>"
    )


def importance_table(model: xgb.Booster, features: list[str]) -> str:
    scores = model.get_score(importance_type="gain")
    rows = sorted([(name, scores.get(name, 0.0)) for name in features], key=lambda item: item[1], reverse=True)[:12]
    max_imp = max([value for _, value in rows] or [1])
    body = []
    for name, value in rows:
        width = value / max_imp * 100 if max_imp else 0
        body.append(
            "<tr>"
            f"<td>{esc(name)}</td>"
            f"<td><div class='bar'><span style='width:{width:.1f}%'></span><b>{fmt(value,3)}</b></div></td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Variable</th><th>Importancia</th></tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def forecast_table(forecast_df: pd.DataFrame) -> str:
    rows = []
    for _, row in forecast_df.head(30).iterrows():
        rows.append(f"<tr><td>{esc(row['fecha'])}</td><td>{fmt(row['horas_paro_pred'],2)}</td></tr>")
    return "<table><thead><tr><th>Fecha</th><th>Horas de paro predichas</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def main() -> None:
    daily = load_daily_data()
    featured = add_features(daily)
    x_data, y_data, features = build_matrix(featured)

    split_idx = int(len(featured) * 0.8)
    train_x, test_x = x_data.iloc[:split_idx], x_data.iloc[split_idx:]
    train_y, test_y = y_data.iloc[:split_idx], y_data.iloc[split_idx:]

    params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "eta": 0.035,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "lambda": 2.0,
        "min_child_weight": 2,
        "seed": 42,
    }
    train_dm = xgb.DMatrix(train_x, label=train_y, feature_names=features)
    test_dm = xgb.DMatrix(test_x, label=test_y, feature_names=features)
    model = xgb.train(params, train_dm, num_boost_round=450, evals=[(test_dm, "validacion")], verbose_eval=False)

    test_pred = np.clip(model.predict(test_dm), 0, None)
    score = metrics(test_y.to_numpy(), test_pred)
    baseline = test_x["media_28d"].to_numpy()
    baseline_score = metrics(test_y.to_numpy(), baseline)

    validation = featured.iloc[split_idx:][["fecha", TARGET]].copy()
    validation["pred"] = test_pred

    forecast_df = forecast(model, daily, features)
    forecast_df.to_csv(FORECAST_OUT, index=False, encoding="utf-8-sig")
    model.save_model(MODEL_OUT)

    metadata = {
        "fuente": SOURCE.name,
        "objetivo": "Predecir horas de paro diarias de Linea 12",
        "rango_datos": {
            "inicio": str(daily["fecha"].min().date()),
            "fin": str(daily["fecha"].max().date()),
            "dias": int(len(daily)),
        },
        "filas_entrenamiento": int(len(train_x)),
        "filas_validacion": int(len(test_x)),
        "features": features,
        "metricas_xgboost": score,
        "metricas_baseline_media_28d": baseline_score,
        "salidas": {
            "modelo": MODEL_OUT.name,
            "pronostico": FORECAST_OUT.name,
            "reporte": REPORT_OUT.name,
        },
    }
    METADATA_OUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    total_pred = float(forecast_df["horas_paro_pred"].sum())
    avg_pred = float(forecast_df["horas_paro_pred"].mean())
    peak = forecast_df.loc[forecast_df["horas_paro_pred"].idxmax()]

    report = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Modelo predictivo de paros L12</title>
  <style>
    :root {{ --bg:#f8fafc; --ink:#0f172a; --muted:#64748b; --line:#dbe3ef; --panel:#fff; --blue:#2563eb; --orange:#f97316; --green:#0f766e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Arial, Helvetica, sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ background:#111827; color:#fff; padding:34px 42px 24px; }}
    header h1 {{ margin:0 0 8px; font-size:32px; letter-spacing:0; }}
    header p {{ margin:0; color:#cbd5e1; line-height:1.45; max-width:980px; }}
    main {{ max-width:1240px; margin:auto; padding:28px 42px 48px; }}
    h2 {{ font-size:20px; margin:0 0 14px; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin-bottom:28px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:28px; }}
    .card, .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
    .card span {{ color:var(--muted); display:block; font-size:13px; margin-bottom:7px; }}
    .card strong {{ display:block; font-size:28px; margin-bottom:5px; }}
    .card small, .note {{ color:var(--muted); line-height:1.45; }}
    .chart {{ background:white; border:1px solid var(--line); border-radius:8px; padding:14px; margin-bottom:28px; overflow-x:auto; }}
    svg {{ width:100%; min-width:820px; height:auto; font-size:12px; fill:#475569; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:10px 9px; text-align:right; }}
    th:first-child, td:first-child {{ text-align:left; }}
    th {{ background:#f1f5f9; color:#334155; }}
    .bar {{ position:relative; height:26px; background:#eef2f7; border-radius:6px; overflow:hidden; min-width:160px; }}
    .bar span {{ position:absolute; inset:0 auto 0 0; background:linear-gradient(90deg,var(--blue),#38bdf8); }}
    .bar b {{ position:relative; display:block; padding:6px 8px; text-align:right; }}
    .pill {{ display:inline-block; padding:4px 8px; border-radius:999px; background:#dcfce7; color:#166534; font-weight:700; font-size:12px; margin-left:8px; }}
    @media (max-width:900px) {{ header, main {{ padding-left:18px; padding-right:18px; }} .metrics, .grid {{ grid-template-columns:1fr; }} header h1 {{ font-size:26px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Modelo predictivo de paros L12 <span class="pill">XGBoost</span></h1>
    <p>Predice horas de paro diarias con base en el histórico del archivo {esc(SOURCE.name)}, usando variables calendario, eventos recientes y rezagos de paro.</p>
  </header>
  <main>
    <section class="metrics">
      <article class="card"><span>MAE validación</span><strong>{fmt(score['mae'],2)} h</strong><small>Error absoluto medio del modelo.</small></article>
      <article class="card"><span>RMSE validación</span><strong>{fmt(score['rmse'],2)} h</strong><small>Penaliza más los días con error alto.</small></article>
      <article class="card"><span>R2 validación</span><strong>{fmt(score['r2'],3)}</strong><small>Comparado contra la variabilidad del periodo de prueba.</small></article>
      <article class="card"><span>Pronóstico 30 días</span><strong>{fmt(total_pred,1)} h</strong><small>Promedio diario {fmt(avg_pred,2)} h; pico el {esc(peak['fecha'])}.</small></article>
    </section>
    <section class="chart">
      <h2>Real reciente, validación y pronóstico</h2>
      {line_svg(daily, validation, forecast_df)}
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Variables más importantes</h2>
        {importance_table(model, features)}
      </div>
      <div class="panel">
        <h2>Pronóstico próximos 30 días</h2>
        {forecast_table(forecast_df)}
      </div>
    </section>
    <section class="panel">
      <h2>Lectura del modelo</h2>
      <p class="note">Elegí predecir horas de paro diarias porque conecta directamente con mantenimiento, capacidad perdida y priorización operativa. El modelo se validó con separación temporal: entrena con el primer 80% de los días disponibles y prueba contra el 20% más reciente. Como referencia, una línea base simple usando media móvil de 28 días tuvo MAE {fmt(baseline_score['mae'],2)} h y RMSE {fmt(baseline_score['rmse'],2)} h.</p>
      <p class="note">El pronóstico es una estimación estadística, no una garantía operativa. Para mejorarlo se pueden agregar variables externas como plan de producción, turnos, mantenimiento preventivo, disponibilidad de materiales, clima, dotación y cambios de presentación programados.</p>
    </section>
  </main>
</body>
</html>
"""
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(REPORT_OUT)
    print(MODEL_OUT)
    print(FORECAST_OUT)


if __name__ == "__main__":
    main()
