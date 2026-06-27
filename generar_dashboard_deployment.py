from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "DP L12 2025 & 2026.xlsx"
OUTPUT = ROOT / "deployment_fallas_2025_vs_2026.html"


def fmt_num(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.{decimals}f}"


def pct(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.{decimals}%}"


def esc(value) -> str:
    return html.escape(str(value))


def period_minutes(series: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    start = pd.to_datetime(series.min()).normalize()
    end = pd.to_datetime(series.max()).normalize() + pd.Timedelta(days=1)
    minutes = max((end - start).total_seconds() / 60, 0)
    return start, end - pd.Timedelta(days=1), minutes


def summarize(data: pd.DataFrame, label: str) -> dict:
    events = len(data)
    downtime_min = float(data["duracion_min"].sum())
    start, end, observed_min = period_minutes(data["Fecha contable"])
    uptime_min = max(observed_min - downtime_min, 0)
    return {
        "label": label,
        "start": start,
        "end": end,
        "events": events,
        "downtime_min": downtime_min,
        "downtime_hr": downtime_min / 60,
        "mttr_min": downtime_min / events if events else 0,
        "mtbf_hr": uptime_min / events / 60 if events else 0,
        "availability": uptime_min / observed_min if observed_min else 0,
        "events_day": events / max((end - start).days + 1, 1),
        "downtime_hr_day": downtime_min / 60 / max((end - start).days + 1, 1),
        "observed_days": max((end - start).days + 1, 1),
    }


def delta(new: float, old: float) -> str:
    if old == 0:
        return "n/a"
    value = (new - old) / old
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1%}"


def bar_table(rows: pd.DataFrame, value_col: str, label_col: str, extra_cols: list[str]) -> str:
    max_value = rows[value_col].max() if not rows.empty else 1
    labels = {"eventos": "Eventos", "mttr_min": "MTTR min", "% horas": "% horas"}
    out = [
        "<table>",
        "<thead><tr><th>Clasificación</th><th>Magnitud</th>"
        + "".join(f"<th>{esc(labels.get(c, c))}</th>" for c in extra_cols)
        + "</tr></thead><tbody>",
    ]
    for _, row in rows.iterrows():
        width = 0 if max_value == 0 else float(row[value_col]) / float(max_value) * 100
        out.append("<tr>")
        out.append(f"<td class='label'>{esc(row[label_col])}</td>")
        out.append(
            "<td><div class='bar-track'>"
            f"<span class='bar' style='width:{width:.1f}%'></span>"
            f"<b>{fmt_num(row[value_col], 1 if value_col.endswith('hr') else 0)}</b>"
            "</div></td>"
        )
        for col in extra_cols:
            val = row[col]
            decimals = 1 if isinstance(val, float) else 0
            out.append(f"<td>{fmt_num(val, decimals) if isinstance(val, (int, float)) else esc(val)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def monthly_svg(monthly: pd.DataFrame) -> str:
    months = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    width, height = 920, 300
    pad_l, pad_r, pad_t, pad_b = 54, 24, 34, 44
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    ymax = max(monthly["horas"].max(), 1)
    group_w = plot_w / 12
    bar_w = group_w * 0.28
    colors = {2025: "#2563eb", 2026: "#f97316"}
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Horas de paro por mes'>"]
    parts.append(f"<line x1='{pad_l}' y1='{height-pad_b}' x2='{width-pad_r}' y2='{height-pad_b}' stroke='#cbd5e1'/>")
    for idx, m in enumerate(range(1, 13)):
        x0 = pad_l + idx * group_w + group_w * 0.25
        parts.append(f"<text x='{pad_l + idx * group_w + group_w/2:.1f}' y='{height-18}' text-anchor='middle'>{months[idx]}</text>")
        for j, year in enumerate([2025, 2026]):
            val = monthly.loc[(monthly["year"] == year) & (monthly["month"] == m), "horas"]
            h = 0 if val.empty else float(val.iloc[0]) / ymax * plot_h
            x = x0 + j * (bar_w + 6)
            y = height - pad_b - h
            parts.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' "
                f"rx='3' fill='{colors[year]}'><title>{year} {months[idx]}: {fmt_num(0 if val.empty else val.iloc[0], 1)} h</title></rect>"
            )
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y = height - pad_b - frac * plot_h
        val = ymax * frac
        parts.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width-pad_r}' y2='{y:.1f}' stroke='#e2e8f0'/>")
        parts.append(f"<text x='{pad_l-10}' y='{y+4:.1f}' text-anchor='end'>{fmt_num(val, 0)}</text>")
    parts.append("<g class='legend'><rect x='690' y='16' width='13' height='13' fill='#2563eb'/><text x='710' y='27'>2025</text>")
    parts.append("<rect x='770' y='16' width='13' height='13' fill='#f97316'/><text x='790' y='27'>2026</text></g>")
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    df = pd.read_excel(SOURCE, sheet_name="Sheet1")
    df["Fecha contable"] = pd.to_datetime(df["Fecha contable"], errors="coerce")
    df = df[df["Fecha contable"].dt.year.isin([2025, 2026])].copy()
    df["year"] = df["Fecha contable"].dt.year
    df["month"] = df["Fecha contable"].dt.month
    df["Tipo de paro"] = df["Tipo de paro"].fillna("Sin clasificar")
    df["Clave de paro"] = df["Clave de paro"].fillna("Sin clasificar")
    df["Subclave de paro"] = df["Subclave de paro"].fillna("Sin clasificar")
    df["duracion_min"] = pd.to_numeric(df["Tiempo formal"], errors="coerce").fillna(
        pd.to_numeric(df["Minutos de paro"], errors="coerce")
    ).fillna(0)

    full_2025 = df[df["year"] == 2025]
    full_2026 = df[df["year"] == 2026]
    ytd_end = full_2026["Fecha contable"].max()
    ytd_2025 = df[(df["year"] == 2025) & (df["Fecha contable"].dt.month < ytd_end.month)]
    ytd_2025 = pd.concat(
        [ytd_2025, df[(df["year"] == 2025) & (df["Fecha contable"].dt.month == ytd_end.month) & (df["Fecha contable"].dt.day <= ytd_end.day)]]
    )

    summaries = [summarize(full_2025, "2025 completo"), summarize(full_2026, "2026 disponible")]
    ytd_summaries = [summarize(ytd_2025, f"2025 YTD a {ytd_end:%d/%m}"), summarize(full_2026, f"2026 YTD a {ytd_end:%d/%m}")]

    by_type = (
        df.groupby(["Tipo de paro", "year"])
        .agg(eventos=("duracion_min", "size"), horas=("duracion_min", lambda s: s.sum() / 60), mttr_min=("duracion_min", "mean"))
        .reset_index()
    )
    type_total = (
        df.groupby("Tipo de paro")
        .agg(eventos=("duracion_min", "size"), horas=("duracion_min", lambda s: s.sum() / 60), mttr_min=("duracion_min", "mean"))
        .reset_index()
        .sort_values("horas", ascending=False)
    )
    top_type = type_total.head(12).copy()
    top_type["% horas"] = (top_type["horas"] / type_total["horas"].sum()).map(lambda value: f"{value:.1%}")

    pivot_type = by_type.pivot(index="Tipo de paro", columns="year", values="horas").fillna(0)
    pivot_type["Total"] = pivot_type.sum(axis=1)
    pivot_type = pivot_type.sort_values("Total", ascending=False).head(10).reset_index()
    for year in [2025, 2026]:
        if year not in pivot_type:
            pivot_type[year] = 0
    pivot_type["Delta 2026 vs 2025"] = pivot_type.apply(lambda r: delta(r[2026], r[2025]), axis=1)

    by_sub = (
        df.groupby("Subclave de paro")
        .agg(eventos=("duracion_min", "size"), horas=("duracion_min", lambda s: s.sum() / 60), mttr_min=("duracion_min", "mean"))
        .reset_index()
        .sort_values("horas", ascending=False)
        .head(12)
    )

    monthly = df.groupby(["year", "month"]).agg(horas=("duracion_min", lambda s: s.sum() / 60)).reset_index()

    s25, s26 = summaries
    y25, y26 = ytd_summaries
    insight = [
        f"En el total disponible, 2026 acumula {fmt_num(s26['downtime_hr'], 1)} h de paro contra {fmt_num(s25['downtime_hr'], 1)} h en 2025; la lectura debe tratarse como parcial porque 2026 llega hasta {s26['end']:%d/%m/%Y}.",
        f"En periodo comparable YTD, las horas de paro cambian {delta(y26['downtime_hr'], y25['downtime_hr'])}, los eventos cambian {delta(y26['events'], y25['events'])} y el MTTR cambia {delta(y26['mttr_min'], y25['mttr_min'])}.",
        f"El tipo dominante por horas es {top_type.iloc[0]['Tipo de paro']} con {fmt_num(top_type.iloc[0]['horas'], 1)} h, equivalente a {top_type.iloc[0]['% horas']} del tiempo formal.",
        f"MTBF se calcula como horas operativas entre eventos: (tiempo calendario observado - paro) / numero de eventos. MTTR es tiempo de paro / numero de eventos.",
    ]

    def metric_cards(items: list[dict]) -> str:
        cards = []
        for item in items:
            cards.append(
                f"<article class='metric'><span>{esc(item['label'])}</span>"
                f"<strong>{fmt_num(item['downtime_hr'], 1)} h</strong>"
                f"<small>{fmt_num(item['events'])} eventos | MTTR {fmt_num(item['mttr_min'], 1)} min | MTBF {fmt_num(item['mtbf_hr'], 2)} h | Disp. {pct(item['availability'], 2)}</small></article>"
            )
        return "".join(cards)

    methodology = {
        "fuente": SOURCE.name,
        "hoja": "Sheet1",
        "duracion": "Tiempo formal; si falta, Minutos de paro",
        "mttr": "Tiempo total de paro / numero de eventos",
        "mtbf": "(Tiempo calendario observado - tiempo de paro) / numero de eventos",
        "periodo_2026": f"{s26['start']:%Y-%m-%d} a {s26['end']:%Y-%m-%d}",
    }

    html_out = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Deployment de fallas L12 | 2025 vs 2026</title>
  <style>
    :root {{
      --bg:#f8fafc; --panel:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#dbe3ef;
      --blue:#2563eb; --orange:#f97316; --green:#0f766e; --red:#b91c1c;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Arial, Helvetica, sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:34px 42px 24px; background:#0f172a; color:white; }}
    header h1 {{ margin:0 0 8px; font-size:32px; letter-spacing:0; }}
    header p {{ margin:0; color:#cbd5e1; max-width:980px; line-height:1.45; }}
    main {{ padding:28px 42px 48px; max-width:1280px; margin:auto; }}
    section {{ margin:0 0 28px; }}
    h2 {{ font-size:20px; margin:0 0 14px; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .metric, .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
    .metric span {{ color:var(--muted); font-size:13px; display:block; margin-bottom:7px; }}
    .metric strong {{ font-size:28px; display:block; margin-bottom:5px; }}
    .metric small {{ color:var(--muted); line-height:1.35; }}
    .insights {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
    .insight {{ border-left:4px solid var(--blue); background:white; border-radius:8px; padding:14px 16px; line-height:1.45; border-top:1px solid var(--line); border-right:1px solid var(--line); border-bottom:1px solid var(--line); }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ padding:10px 9px; border-bottom:1px solid var(--line); text-align:right; vertical-align:middle; }}
    th:first-child, td:first-child {{ text-align:left; }}
    th {{ color:#334155; background:#f1f5f9; font-weight:700; }}
    td.label {{ max-width:280px; }}
    .bar-track {{ position:relative; height:26px; background:#eef2f7; border-radius:6px; overflow:hidden; min-width:170px; }}
    .bar {{ position:absolute; inset:0 auto 0 0; background:linear-gradient(90deg,var(--blue),#38bdf8); }}
    .bar-track b {{ position:relative; display:block; padding:6px 8px; color:#0f172a; text-align:right; }}
    .chart {{ background:white; border:1px solid var(--line); border-radius:8px; padding:14px; overflow-x:auto; }}
    svg {{ width:100%; min-width:760px; height:auto; font-size:12px; fill:#475569; }}
    .comparison table td:nth-child(2), .comparison table td:nth-child(3) {{ font-weight:700; }}
    .method {{ color:var(--muted); font-size:12px; line-height:1.5; }}
    .pill {{ display:inline-block; padding:4px 8px; border-radius:999px; background:#e0f2fe; color:#075985; font-weight:700; font-size:12px; margin-left:8px; }}
    @media (max-width:900px) {{ header, main {{ padding-left:18px; padding-right:18px; }} .metrics, .grid, .insights {{ grid-template-columns:1fr; }} header h1 {{ font-size:26px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Deployment de fallas L12 <span class="pill">2025 vs 2026</span></h1>
    <p>Dashboard generado desde el archivo {esc(SOURCE.name)}. Clasifica los paros por tipo de falla y compara volumen, tiempo de paro, MTTR, MTBF y disponibilidad operacional.</p>
  </header>
  <main>
    <section>
      <h2>Resumen ejecutivo</h2>
      <div class="insights">{''.join(f"<div class='insight'>{esc(x)}</div>" for x in insight)}</div>
    </section>
    <section>
      <h2>KPIs por periodo</h2>
      <div class="metrics">{metric_cards(summaries + ytd_summaries)}</div>
    </section>
    <section>
      <h2>Horas de paro por mes</h2>
      <div class="chart">{monthly_svg(monthly)}</div>
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Clasificación por tipo de falla</h2>
        {bar_table(top_type, "horas", "Tipo de paro", ["eventos", "mttr_min", "% horas"])}
      </div>
      <div class="panel">
        <h2>Subclaves principales</h2>
        {bar_table(by_sub, "horas", "Subclave de paro", ["eventos", "mttr_min"])}
      </div>
    </section>
    <section class="panel comparison">
      <h2>Comparativo por tipo: horas 2025 vs 2026 disponible</h2>
      <table>
        <thead><tr><th>Tipo de falla</th><th>2025 h</th><th>2026 h</th><th>Delta</th><th>Total h</th></tr></thead>
        <tbody>
          {''.join(f"<tr><td>{esc(r['Tipo de paro'])}</td><td>{fmt_num(r[2025],1)}</td><td>{fmt_num(r[2026],1)}</td><td>{esc(r['Delta 2026 vs 2025'])}</td><td>{fmt_num(r['Total'],1)}</td></tr>" for _, r in pivot_type.iterrows())}
        </tbody>
      </table>
    </section>
    <section class="panel">
      <h2>Metodología de mantenimiento</h2>
      <p class="method">MTTR: tiempo medio de reparación o recuperación por evento. MTBF: tiempo medio entre fallas, calculado sobre tiempo operativo observado. La disponibilidad se estima como tiempo operativo / tiempo calendario observado. Para 2026 el archivo contiene datos hasta {s26['end']:%d/%m/%Y}; por eso se incluye comparativo YTD contra 2025 al mismo día y mes.</p>
      <pre class="method">{esc(json.dumps(methodology, ensure_ascii=False, indent=2))}</pre>
    </section>
  </main>
</body>
</html>
"""
    OUTPUT.write_text(html_out, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
