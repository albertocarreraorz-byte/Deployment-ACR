from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "DP L12 2025 & 2026.xlsx"
OUT_DIR = ROOT / "modelos_tipos_paro"
REPORT_OUT = ROOT / "modelos_prediccion_tipos_paro.html"
FORECAST_OUT = ROOT / "pronostico_tipos_paro_30_dias.csv"
METADATA_OUT = ROOT / "modelos_tipos_paro_metadata.json"

TIPOS_OBJETIVO = [
    "PERDIDA VELOCIDAD Y MICROPAROS",
    "P. OTROS DEPTOS.",
    "P. POR EQUIPO",
]
FORECAST_DAYS = 30
LAGS = [1, 2, 3, 7, 14, 28]
ROLLS = [7, 14, 28]
TOP_SUBCLAVES = 6


def esc(value) -> str:
    return html.escape(str(value))


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return cleaned or "tipo_paro"


def fmt(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.{decimals}f}"


def load_source() -> pd.DataFrame:
    df = pd.read_excel(SOURCE, sheet_name="Sheet1")
    df["Fecha contable"] = pd.to_datetime(df["Fecha contable"], errors="coerce")
    df = df.dropna(subset=["Fecha contable"]).copy()
    df["Tipo de paro"] = df["Tipo de paro"].fillna("Sin clasificar").astype(str).str.strip()
    df["Clave de paro"] = df["Clave de paro"].fillna("Sin clasificar").astype(str).str.strip()
    df["Clave1 de paro"] = df["Clave1 de paro"].fillna("Sin clasificar").astype(str).str.strip()
    df["Subclave de paro"] = df["Subclave de paro"].fillna("Sin clasificar").astype(str).str.strip()
    df["tiempo_min"] = pd.to_numeric(df["Tiempo formal"], errors="coerce").fillna(
        pd.to_numeric(df["Minutos de paro"], errors="coerce")
    )
    df["tiempo_min"] = df["tiempo_min"].fillna(0).clip(lower=0)
    df["fecha"] = df["Fecha contable"].dt.normalize()
    return df


def add_calendar(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["dia_semana"] = out["fecha"].dt.dayofweek
    out["mes"] = out["fecha"].dt.month
    out["dia_mes"] = out["fecha"].dt.day
    out["semana_anio"] = out["fecha"].dt.isocalendar().week.astype(int)
    out["trimestre"] = out["fecha"].dt.quarter
    out["es_fin_semana"] = (out["dia_semana"] >= 5).astype(int)
    out["dias_desde_inicio"] = (out["fecha"] - out["fecha"].min()).dt.days
    out["sin_dia"] = np.sin(2 * np.pi * out["dia_semana"] / 7)
    out["cos_dia"] = np.cos(2 * np.pi * out["dia_semana"] / 7)
    out["sin_mes"] = np.sin(2 * np.pi * out["mes"] / 12)
    out["cos_mes"] = np.cos(2 * np.pi * out["mes"] / 12)
    return out


def build_daily_for_type(df: pd.DataFrame, tipo: str) -> tuple[pd.DataFrame, list[str]]:
    tipo_df = df[df["Tipo de paro"] == tipo].copy()
    full_dates = pd.DataFrame({"fecha": pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")})

    daily = (
        tipo_df.groupby("fecha")
        .agg(
            minutos_objetivo=("tiempo_min", "sum"),
            eventos_objetivo=("tiempo_min", "size"),
            mttr_objetivo=("tiempo_min", "mean"),
        )
        .reset_index()
    )
    daily = full_dates.merge(daily, on="fecha", how="left")
    daily[["minutos_objetivo", "eventos_objetivo", "mttr_objetivo"]] = daily[
        ["minutos_objetivo", "eventos_objetivo", "mttr_objetivo"]
    ].fillna(0)

    top_subclaves = (
        tipo_df.groupby("Subclave de paro")["tiempo_min"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_SUBCLAVES)
        .index.tolist()
    )

    for idx, subclave in enumerate(top_subclaves, start=1):
        col_base = f"subclave_{idx}_min"
        sub_daily = (
            tipo_df[tipo_df["Subclave de paro"] == subclave]
            .groupby("fecha")["tiempo_min"]
            .sum()
            .rename(col_base)
            .reset_index()
        )
        daily = daily.merge(sub_daily, on="fecha", how="left")
        daily[col_base] = daily[col_base].fillna(0)

    daily = add_calendar(daily)
    for lag in LAGS:
        daily[f"target_lag_{lag}d"] = daily["minutos_objetivo"].shift(lag)
        daily[f"eventos_lag_{lag}d"] = daily["eventos_objetivo"].shift(lag)

    for window in ROLLS:
        shifted_target = daily["minutos_objetivo"].shift(1)
        shifted_events = daily["eventos_objetivo"].shift(1)
        daily[f"target_media_{window}d"] = shifted_target.rolling(window, min_periods=1).mean()
        daily[f"target_max_{window}d"] = shifted_target.rolling(window, min_periods=1).max()
        daily[f"target_suma_{window}d"] = shifted_target.rolling(window, min_periods=1).sum()
        daily[f"eventos_media_{window}d"] = shifted_events.rolling(window, min_periods=1).mean()

    for idx in range(1, len(top_subclaves) + 1):
        col_base = f"subclave_{idx}_min"
        daily[f"{col_base}_lag_1d"] = daily[col_base].shift(1)
        daily[f"{col_base}_media_7d"] = daily[col_base].shift(1).rolling(7, min_periods=1).mean()
        daily[f"{col_base}_suma_28d"] = daily[col_base].shift(1).rolling(28, min_periods=1).sum()

    feature_cols = [
        col
        for col in daily.columns
        if col
        not in {
            "fecha",
            "minutos_objetivo",
            "eventos_objetivo",
            "mttr_objetivo",
            *[f"subclave_{idx}_min" for idx in range(1, len(top_subclaves) + 1)],
        }
    ]
    daily = daily.dropna(subset=feature_cols).reset_index(drop=True)
    return daily, top_subclaves


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(error) / denom) * 100)
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0
    return {"mae_min": mae, "rmse_min": rmse, "mape_pct": mape, "r2": float(r2)}


def train_model(daily: pd.DataFrame, tipo: str, top_subclaves: list[str]) -> dict:
    excluded_prefix = tuple(f"subclave_{idx}_min" for idx in range(1, len(top_subclaves) + 1))
    features = [
        col
        for col in daily.columns
        if col not in {"fecha", "minutos_objetivo", "eventos_objetivo", "mttr_objetivo"}
        and not (col.startswith(excluded_prefix) and not any(col.endswith(suffix) for suffix in ["lag_1d", "media_7d", "suma_28d"]))
    ]
    x_data = daily[features]
    y_data = daily["minutos_objetivo"]
    split_idx = int(len(daily) * 0.8)
    train_x, test_x = x_data.iloc[:split_idx], x_data.iloc[split_idx:]
    train_y, test_y = y_data.iloc[:split_idx], y_data.iloc[split_idx:]

    params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "eta": 0.04,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "lambda": 2.5,
        "min_child_weight": 2,
        "seed": 42,
    }
    train_dm = xgb.DMatrix(train_x, label=train_y, feature_names=features)
    test_dm = xgb.DMatrix(test_x, label=test_y, feature_names=features)

    model_raw = xgb.train(params, train_dm, num_boost_round=420, evals=[(test_dm, "validacion")], verbose_eval=False)
    pred_raw = np.clip(model_raw.predict(test_dm), 0, None)

    train_dm_log = xgb.DMatrix(train_x, label=np.log1p(train_y), feature_names=features)
    model_log = xgb.train(params, train_dm_log, num_boost_round=420, evals=[(test_dm, "validacion")], verbose_eval=False)
    pred_log = np.clip(np.expm1(model_log.predict(test_dm)), 0, None)

    candidates = {
        "xgboost_minutos": pred_raw,
        "xgboost_log_minutos": pred_log,
        "media_movil_7d": test_x["target_media_7d"].to_numpy(),
        "media_movil_14d": test_x["target_media_14d"].to_numpy(),
        "media_movil_28d": test_x["target_media_28d"].to_numpy(),
    }
    candidate_metrics = {name: metrics(test_y.to_numpy(), pred) for name, pred in candidates.items()}
    selected_kind = min(candidate_metrics, key=lambda name: candidate_metrics[name]["mae_min"])
    pred = np.clip(candidates[selected_kind], 0, None)

    slug = safe_name(tipo)
    raw_model_path = OUT_DIR / f"modelo_{slug}_xgboost_minutos.json"
    log_model_path = OUT_DIR / f"modelo_{slug}_xgboost_log_minutos.json"
    selector_path = OUT_DIR / f"modelo_{slug}_seleccionado.json"
    model_raw.save_model(raw_model_path)
    model_log.save_model(log_model_path)
    selector_path.write_text(
        json.dumps(
            {
                "tipo_paro": tipo,
                "modelo_seleccionado": selected_kind,
                "criterio": "menor MAE en validacion temporal",
                "candidatos": candidate_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    validation = daily.iloc[split_idx:][["fecha", "minutos_objetivo"]].copy()
    validation["prediccion_min"] = pred
    validation["tipo_paro"] = tipo

    return {
        "tipo": tipo,
        "slug": slug,
        "model_raw": model_raw,
        "model_log": model_log,
        "model_path": selector_path,
        "raw_model_path": raw_model_path,
        "log_model_path": log_model_path,
        "selected_kind": selected_kind,
        "features": features,
        "top_subclaves": top_subclaves,
        "daily": daily,
        "validation": validation,
        "metrics": candidate_metrics[selected_kind],
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": candidate_metrics["media_movil_28d"],
        "split_idx": split_idx,
    }


def forecast_type(result: dict, source_df: pd.DataFrame) -> pd.DataFrame:
    tipo = result["tipo"]
    top_subclaves = result["top_subclaves"]
    selected_kind = result["selected_kind"]
    features = result["features"]

    history, _ = build_daily_for_type(source_df, tipo)
    raw_history = history[
        ["fecha", "minutos_objetivo", "eventos_objetivo", "mttr_objetivo"]
        + [f"subclave_{idx}_min" for idx in range(1, len(top_subclaves) + 1)]
    ].copy()

    preds = []
    for _ in range(FORECAST_DAYS):
        next_date = raw_history["fecha"].max() + pd.Timedelta(days=1)
        row = {
            "fecha": next_date,
            "minutos_objetivo": np.nan,
            "eventos_objetivo": 0,
            "mttr_objetivo": 0,
        }
        for idx in range(1, len(top_subclaves) + 1):
            row[f"subclave_{idx}_min"] = 0
        candidate = pd.concat([raw_history, pd.DataFrame([row])], ignore_index=True)

        candidate = add_calendar(candidate)
        for lag in LAGS:
            candidate[f"target_lag_{lag}d"] = candidate["minutos_objetivo"].shift(lag)
            candidate[f"eventos_lag_{lag}d"] = candidate["eventos_objetivo"].shift(lag)
        for window in ROLLS:
            shifted_target = candidate["minutos_objetivo"].shift(1)
            shifted_events = candidate["eventos_objetivo"].shift(1)
            candidate[f"target_media_{window}d"] = shifted_target.rolling(window, min_periods=1).mean()
            candidate[f"target_max_{window}d"] = shifted_target.rolling(window, min_periods=1).max()
            candidate[f"target_suma_{window}d"] = shifted_target.rolling(window, min_periods=1).sum()
            candidate[f"eventos_media_{window}d"] = shifted_events.rolling(window, min_periods=1).mean()
        for idx in range(1, len(top_subclaves) + 1):
            base = f"subclave_{idx}_min"
            candidate[f"{base}_lag_1d"] = candidate[base].shift(1)
            candidate[f"{base}_media_7d"] = candidate[base].shift(1).rolling(7, min_periods=1).mean()
            candidate[f"{base}_suma_28d"] = candidate[base].shift(1).rolling(28, min_periods=1).sum()

        feature_row = candidate[candidate["fecha"] == next_date]
        if selected_kind == "xgboost_minutos":
            pred = float(result["model_raw"].predict(xgb.DMatrix(feature_row[features], feature_names=features))[0])
        elif selected_kind == "xgboost_log_minutos":
            pred = float(np.expm1(result["model_log"].predict(xgb.DMatrix(feature_row[features], feature_names=features))[0]))
        elif selected_kind == "media_movil_7d":
            pred = float(feature_row["target_media_7d"].iloc[0])
        elif selected_kind == "media_movil_14d":
            pred = float(feature_row["target_media_14d"].iloc[0])
        else:
            pred = float(feature_row["target_media_28d"].iloc[0])
        pred = max(pred, 0)
        preds.append({"fecha": next_date.date(), "tipo_paro": tipo, "minutos_predichos": pred, "horas_predichas": pred / 60})
        raw_history.loc[len(raw_history)] = {
            **row,
            "minutos_objetivo": pred,
        }

    return pd.DataFrame(preds)


def make_bar(value: float, max_value: float) -> str:
    width = 0 if max_value <= 0 else value / max_value * 100
    return f"<div class='bar'><span style='width:{width:.1f}%'></span><b>{fmt(value, 1)}</b></div>"


def report_html(results: list[dict], forecasts: pd.DataFrame) -> str:
    cards = []
    for result in results:
        fc = forecasts[forecasts["tipo_paro"] == result["tipo"]]
        cards.append(
            "<article class='card'>"
            f"<span>{esc(result['tipo'])}</span>"
            f"<strong>{fmt(fc['minutos_predichos'].sum(), 0)} min</strong>"
            f"<small>Modelo: {esc(result['selected_kind'])} | 30 dias: {fmt(fc['horas_predichas'].sum(), 1)} h | MAE {fmt(result['metrics']['mae_min'], 1)} min</small>"
            "</article>"
        )

    metric_rows = []
    for result in results:
        metric_rows.append(
            "<tr>"
            f"<td>{esc(result['tipo'])}</td>"
            f"<td>{esc(result['selected_kind'])}</td>"
            f"<td>{fmt(result['metrics']['mae_min'], 1)}</td>"
            f"<td>{fmt(result['metrics']['rmse_min'], 1)}</td>"
            f"<td>{fmt(result['metrics']['mape_pct'], 1)}%</td>"
            f"<td>{fmt(result['metrics']['r2'], 3)}</td>"
            f"<td>{fmt(result['baseline_metrics']['mae_min'], 1)}</td>"
            "</tr>"
        )

    forecast_rows = []
    max_forecast = forecasts["minutos_predichos"].max()
    for _, row in forecasts.iterrows():
        forecast_rows.append(
            "<tr>"
            f"<td>{esc(row['fecha'])}</td>"
            f"<td>{esc(row['tipo_paro'])}</td>"
            f"<td>{make_bar(float(row['minutos_predichos']), float(max_forecast))}</td>"
            f"<td>{fmt(row['horas_predichas'], 2)}</td>"
            "</tr>"
        )

    subclave_blocks = []
    for result in results:
        items = "".join(f"<li>{esc(item)}</li>" for item in result["top_subclaves"])
        subclave_blocks.append(f"<div class='panel'><h3>{esc(result['tipo'])}</h3><ul>{items}</ul></div>")

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Modelos por tipo de paro L12</title>
  <style>
    :root {{ --bg:#f8fafc; --panel:#fff; --ink:#0f172a; --muted:#64748b; --line:#dbe3ef; --blue:#2563eb; --green:#0f766e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Arial, Helvetica, sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ background:#111827; color:#fff; padding:34px 42px 24px; }}
    header h1 {{ margin:0 0 8px; font-size:32px; letter-spacing:0; }}
    header p {{ margin:0; color:#cbd5e1; line-height:1.45; max-width:1020px; }}
    main {{ max-width:1280px; margin:auto; padding:28px 42px 48px; }}
    h2 {{ font-size:20px; margin:0 0 14px; }}
    h3 {{ margin:0 0 10px; font-size:15px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-bottom:28px; }}
    .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-bottom:28px; }}
    .card, .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
    .card span {{ color:var(--muted); display:block; font-size:13px; margin-bottom:7px; min-height:32px; }}
    .card strong {{ display:block; font-size:28px; margin-bottom:5px; }}
    .card small, .note {{ color:var(--muted); line-height:1.45; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; background:white; }}
    th, td {{ border-bottom:1px solid var(--line); padding:10px 9px; text-align:right; vertical-align:middle; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
    th {{ background:#f1f5f9; color:#334155; }}
    .bar {{ position:relative; height:26px; background:#eef2f7; border-radius:6px; overflow:hidden; min-width:180px; }}
    .bar span {{ position:absolute; inset:0 auto 0 0; background:linear-gradient(90deg,var(--blue),#38bdf8); }}
    .bar b {{ position:relative; display:block; padding:6px 8px; text-align:right; color:#0f172a; }}
    ul {{ margin:0; padding-left:18px; color:#334155; line-height:1.55; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; margin-bottom:28px; }}
    @media (max-width:900px) {{ header, main {{ padding-left:18px; padding-right:18px; }} .metrics, .grid {{ grid-template-columns:1fr; }} header h1 {{ font-size:26px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Modelos predictivos por tipo de paro L12</h1>
      <p>Se entrenaron modelos independientes para predecir minutos diarios de Tiempo formal por tipo de paro. Cada tipo compara XGBoost contra medias moviles y conserva el candidato con menor error en validacion temporal.</p>
  </header>
  <main>
    <section class="metrics">{''.join(cards)}</section>
    <section>
      <h2>Metricas de validacion temporal</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>Tipo de paro</th><th>Modelo seleccionado</th><th>MAE min</th><th>RMSE min</th><th>MAPE</th><th>R2</th><th>MAE baseline 28d</th></tr></thead>
        <tbody>{''.join(metric_rows)}</tbody>
      </table></div>
    </section>
    <section>
      <h2>Pronostico por dia y tipo de paro</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>Fecha</th><th>Tipo de paro</th><th>Minutos predichos</th><th>Horas</th></tr></thead>
        <tbody>{''.join(forecast_rows)}</tbody>
      </table></div>
    </section>
    <section>
      <h2>Subclaves historicas usadas como senales</h2>
      <div class="grid">{''.join(subclave_blocks)}</div>
    </section>
    <section class="panel">
      <h2>Metodo</h2>
      <p class="note">El objetivo de cada modelo es la suma diaria de <b>Tiempo formal</b> en minutos para su tipo de paro. Se usaron variables calendario, rezagos del propio tipo de paro, medias moviles y senales rezagadas de las subclaves principales. La validacion respeta el tiempo: el primer 80% de dias entrena y el 20% mas reciente prueba. Para pronostico, las subclaves futuras desconocidas se tratan como cero y se usan solo sus rezagos historicos.</p>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load_source()

    results = []
    forecasts = []
    for tipo in TIPOS_OBJETIVO:
        daily, top_subclaves = build_daily_for_type(df, tipo)
        result = train_model(daily, tipo, top_subclaves)
        results.append(result)
        forecasts.append(forecast_type(result, df))

    forecast_df = pd.concat(forecasts, ignore_index=True)
    forecast_df.to_csv(FORECAST_OUT, index=False, encoding="utf-8-sig")

    metadata = {
        "fuente": SOURCE.name,
        "objetivo": "Prediccion de minutos diarios de Tiempo formal por tipo de paro",
        "tipos_modelados": TIPOS_OBJETIVO,
        "rango_datos": {
            "inicio": str(df["fecha"].min().date()),
            "fin": str(df["fecha"].max().date()),
            "dias": int((df["fecha"].max() - df["fecha"].min()).days + 1),
        },
        "modelos": [
            {
                "tipo": result["tipo"],
                "archivo_modelo": str(result["model_path"].relative_to(ROOT)),
                "modelo_seleccionado": result["selected_kind"],
                "archivo_xgboost_minutos": str(result["raw_model_path"].relative_to(ROOT)),
                "archivo_xgboost_log_minutos": str(result["log_model_path"].relative_to(ROOT)),
                "metricas": result["metrics"],
                "metricas_candidatos": result["candidate_metrics"],
                "baseline_media_28d": result["baseline_metrics"],
                "top_subclaves": result["top_subclaves"],
                "features": result["features"],
            }
            for result in results
        ],
        "salidas": {
            "reporte": REPORT_OUT.name,
            "pronostico": FORECAST_OUT.name,
            "metadata": METADATA_OUT.name,
        },
    }
    METADATA_OUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_OUT.write_text(report_html(results, forecast_df), encoding="utf-8")

    print(REPORT_OUT)
    print(FORECAST_OUT)
    print(METADATA_OUT)
    for result in results:
        print(result["model_path"])


if __name__ == "__main__":
    main()
