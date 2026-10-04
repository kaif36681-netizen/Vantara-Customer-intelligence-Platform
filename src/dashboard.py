"""Interactive results dashboard (single HTML file, opens in your browser).

Usage:
    python -m src.dashboard            # build reports/dashboard.html and open it
    python -m src.dashboard --no-open  # build only

Needs the outputs of `src.pipeline` and `src.models.train_baselines`. No extra packages
and no internet connection are required (charts are drawn with plain SVG/JS).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src.models.train_baselines import split_data
from src.utils.config import load_config, resolve_path
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

NICE = {"recency_days": "Days since last order", "frequency": "Number of orders",
        "monetary_total": "Total spend", "monetary_avg": "Avg order value",
        "avg_basket_size": "Avg basket size", "tenure_days": "Customer age (days)",
        "gap_variance": "Irregularity of orders", "frequency_trend": "Order-frequency trend",
        "seasonal_concentration": "Seasonality", "return_rate": "Return rate",
        "n_return_lines": "Returned items", "clv_historical": "Net lifetime spend",
        "discount_sensitivity": "Discount sensitivity", "n_unique_products": "Product variety",
        "engagement_score": "Engagement score", "is_uk": "UK customer"}


def _label(name: str) -> str:
    return NICE.get(name, name.replace("cat_share_", "Category: ").replace("_", " ").title())


def collect(cfg: dict) -> dict:
    """Gather every number the dashboard shows."""
    art = resolve_path(cfg["paths"]["artifacts"])
    bundle = joblib.load(art / "churn_best.joblib")
    model, feats, thr = bundle["model"], bundle["features"], bundle["threshold"]
    comp = pd.read_csv(art / "model_comparison.csv")
    df = pd.read_csv(resolve_path(cfg["paths"]["processed"]))

    _, _, X_te, _, _, y_te = split_data(df, cfg)
    p_te = model.predict_proba(X_te[feats])[:, 1]
    pred = (p_te >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_te, p_te)
    step = max(len(fpr) // 120, 1)

    imp = permutation_importance(model, X_te[feats], y_te, scoring="roc_auc", n_repeats=5,
                                 random_state=cfg["seed"])
    imp = pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False).head(8)

    df["p"] = model.predict_proba(df[feats])[:, 1]
    risk = pd.cut(df["p"], [-1, thr - 0.2, thr - 1e-9, 2], labels=["Low", "Medium", "High"])
    top = (df[df["p"] >= thr].sort_values("clv_historical", ascending=False).head(12))
    best_row = comp.iloc[0]
    return {
        "best": str(best_row["model"]), "threshold": round(float(thr), 3),
        "cutoff": cfg["churn"]["cutoff_date"], "window": cfg["churn"]["window_days"],
        "kpi": {"customers": int(len(df)), "churn_rate": float(df["churned"].mean()),
                "revenue_at_risk": float(df.loc[df["p"] >= thr, "clv_historical"].sum()),
                "auc": float(roc_auc_score(y_te, p_te)), "accuracy": float(accuracy_score(y_te, pred)),
                "precision": float(precision_score(y_te, pred, zero_division=0)),
                "recall": float(recall_score(y_te, pred, zero_division=0)),
                "f1": float(f1_score(y_te, pred, zero_division=0))},
        "models": [{"name": r.model, "cv": r.cv_auc, "val": r.val_auc, "test": r.test_roc_auc,
                    "recall": r.test_recall, "precision": r.test_precision}
                   for r in comp.itertuples()],
        "roc": {"fpr": [round(float(v), 4) for v in fpr[::step]], "tpr": [round(float(v), 4) for v in tpr[::step]]},
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "importance": [{"name": _label(k), "value": float(v)} for k, v in imp.items()],
        "risk": {k: int((risk == k).sum()) for k in ["High", "Medium", "Low"]},
        "top": [{"id": int(r.customer_id), "p": float(r.p), "value": float(r.clv_historical),
                 "recency": float(r.recency_days), "orders": int(r.frequency)} for r in top.itertuples()],
    }


def open_in_browser(path: Path) -> None:
    """Open the file in a real browser. On Windows, .html is sometimes associated with
    Notepad, so Edge/Chrome/Firefox are launched directly when found."""
    url = path.resolve().as_uri()
    if sys.platform.startswith("win"):
        bases = [os.environ.get(k) for k in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
        rel = ["Microsoft/Edge/Application/msedge.exe", "Google/Chrome/Application/chrome.exe",
               "Mozilla Firefox/firefox.exe"]
        for base in filter(None, bases):
            for r in rel:
                exe = Path(base) / r
                if exe.exists():
                    subprocess.Popen([str(exe), url])
                    return
    webbrowser.open(url)


def build_dashboard(config_path: str | None = None, open_browser: bool = True):
    """Write reports/dashboard.html and (optionally) open it."""
    cfg = load_config(config_path)
    data = collect(cfg)
    html = TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    out = resolve_path("reports") / "dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    log.info("Dashboard written to %s", out)
    print(f"\nDashboard file: {out}")
    if open_browser:
        open_in_browser(out)
    return out


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Customer Churn Dashboard</title><style>
:root{--bg:#0f1420;--card:#171e2e;--ink:#e8ecf5;--mute:#8a94ab;--line:#26304a;--a:#6ea8ff;--b:#35d0a0;--c:#ffb454;--d:#ff6b81;--r:14px}
@media (prefers-color-scheme:light){:root{--bg:#f2f5fb;--card:#fff;--ink:#16203a;--mute:#66708a;--line:#e1e6f2;--a:#2f6fe4;--b:#0f9d76;--c:#d98a1f;--d:#e0354f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,-apple-system,Arial,sans-serif}
header{padding:28px 32px 8px}h1{margin:0;font-size:26px}header p{margin:4px 0 0;color:var(--mute)}
main{padding:16px 32px 40px;display:grid;gap:18px;grid-template-columns:repeat(12,1fr)}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:18px 20px;min-width:0}
.card h2{margin:0 0 12px;font-size:15px;color:var(--mute);font-weight:600;letter-spacing:.03em;text-transform:uppercase}
.kpi{grid-column:span 3}.kpi .v{font-size:32px;font-weight:700}.kpi .s{color:var(--mute);font-size:13px}
.w4{grid-column:span 4}.w5{grid-column:span 5}.w6{grid-column:span 6}.w7{grid-column:span 7}.w8{grid-column:span 8}.w12{grid-column:span 12}
@media(max-width:1000px){.kpi{grid-column:span 6}.w4,.w5,.w6,.w7,.w8{grid-column:span 12}}
.row{display:grid;grid-template-columns:150px 1fr 54px;gap:10px;align-items:center;margin:7px 0;font-size:13px}
.track{background:var(--line);border-radius:8px;height:10px;overflow:hidden}.fill{height:100%;border-radius:8px}
.best{color:var(--b);font-weight:700}.legend{display:flex;gap:16px;font-size:12px;color:var(--mute);margin-bottom:6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
.cm{display:grid;grid-template-columns:auto 1fr 1fr;gap:6px;text-align:center;font-size:13px}.cm div{padding:16px 6px;border-radius:10px}
.cm .h{color:var(--mute);padding:6px}.ok{background:color-mix(in srgb,var(--b) 25%,transparent)}.bad{background:color-mix(in srgb,var(--d) 22%,transparent)}.cm b{display:block;font-size:24px}
table{width:100%;border-collapse:collapse;font-size:14px}th{color:var(--mute);text-align:left;font-weight:600;padding:8px;border-bottom:1px solid var(--line)}td{padding:9px 8px;border-bottom:1px solid var(--line)}
.pill{padding:2px 10px;border-radius:99px;background:color-mix(in srgb,var(--d) 22%,transparent);color:var(--d);font-weight:600}
.note{color:var(--mute);font-size:12.5px;margin-top:10px}
</style></head><body>
<header><h1>Customer Churn Prediction Dashboard</h1><p id="sub"></p></header>
<main id="m"></main>
<script>
const D=__DATA__, $=s=>document.querySelector(s), pct=v=>(v*100).toFixed(1)+'%', money=v=>'£'+Math.round(v).toLocaleString();
const nice=n=>n.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
$('#sub').textContent=`Will a customer stay silent for ${D.window} days after ${D.cutoff}? Best model: ${nice(D.best)} · decision threshold ${D.threshold}`;
const card=(cls,title,body)=>`<section class="card ${cls}"><h2>${title}</h2>${body}</section>`;
const kpi=(t,v,s,c)=>card('kpi',t,`<div class="v" style="color:var(--${c})">${v}</div><div class="s">${s}</div>`);
const bar=(label,v,max,col,txt)=>`<div class="row"><span>${label}</span><div class="track"><div class="fill" style="width:${Math.max(v/max*100,1)}%;background:var(--${col})"></div></div><span>${txt}</span></div>`;
let h='';
h+=kpi('Customers analysed',D.kpi.customers.toLocaleString(),`${pct(D.kpi.churn_rate)} churned in the window`,'a');
h+=kpi('Test ROC-AUC',D.kpi.auc.toFixed(3),'ranking quality on unseen customers','b');
h+=kpi('Churners caught (recall)',pct(D.kpi.recall),`precision ${pct(D.kpi.precision)} · accuracy ${pct(D.kpi.accuracy)} · F1 ${D.kpi.f1.toFixed(2)}`,'c');
h+=kpi('Revenue at risk',money(D.kpi.revenue_at_risk),'past spend of customers flagged high-risk','d');
// model comparison
const mx=Math.max(...D.models.map(m=>Math.max(m.val,m.test)),.01);
h+=card('w6','Model comparison (ROC-AUC)',`<div class="legend"><span><i style="background:var(--a)"></i>Validation</span><span><i style="background:var(--b)"></i>Test</span></div>`+
 D.models.map(m=>`<div style="margin-bottom:6px"><div class="${m.name==D.best?'best':''}" style="font-size:13px">${nice(m.name)}${m.name==D.best?' ★ selected':''}</div>`+bar('Validation',m.val-.5,mx-.5,'a',m.val.toFixed(3))+bar('Test',m.test-.5,mx-.5,'b',m.test.toFixed(3))+`</div>`).join('')+
 `<div class="note">Bars start at 0.5 (random guessing). The model is selected on validation AUC, never on test.</div>`);
// ROC
const W=300,H=240,P=30,x=v=>P+v*(W-P-8),y=v=>H-P-v*(H-P-8);
const pts=D.roc.fpr.map((f,i)=>`${x(f)},${y(D.roc.tpr[i])}`).join(' ');
h+=card('w6','ROC curve (best model, test set)',`<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-height:290px">
 <line x1="${P}" y1="${y(0)}" x2="${x(1)}" y2="${y(0)}" stroke="var(--line)"/><line x1="${P}" y1="${y(0)}" x2="${P}" y2="${y(1)}" stroke="var(--line)"/>
 <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="var(--mute)" stroke-dasharray="4 4"/>
 <polygon points="${x(0)},${y(0)} ${pts} ${x(1)},${y(0)}" fill="var(--a)" opacity=".16"/><polyline points="${pts}" fill="none" stroke="var(--a)" stroke-width="2.5"/>
 <text x="${W/2}" y="${H-6}" fill="var(--mute)" font-size="10" text-anchor="middle">False positive rate</text>
 <text x="10" y="${H/2}" fill="var(--mute)" font-size="10" transform="rotate(-90 10 ${H/2})" text-anchor="middle">True positive rate</text>
 <text x="${x(.55)}" y="${y(.3)}" fill="var(--a)" font-size="14" font-weight="700">AUC ${D.kpi.auc.toFixed(3)}</text></svg>`);
// importance
const im=Math.max(...D.importance.map(i=>i.value),1e-6);
h+=card('w4','What drives churn (permutation importance)',D.importance.map(i=>bar(i.name,Math.max(i.value,0),im,'c',i.value.toFixed(3))).join('')+`<div class="note">Drop in test AUC when that feature is shuffled; bigger = more important.</div>`);
// donut
const tot=Object.values(D.risk).reduce((a,b)=>a+b,0),cols={High:'d',Medium:'c',Low:'b'};let off=0;
const arcs=['High','Medium','Low'].map(k=>{const f=D.risk[k]/tot,a=`<circle r="15.9" cx="21" cy="21" fill="none" stroke="var(--${cols[k]})" stroke-width="6" stroke-dasharray="${f*100} ${100-f*100}" stroke-dashoffset="${-off*100}" transform="rotate(-90 21 21)"/>`;off+=f;return a}).join('');
h+=card('w4','Customer risk segments',`<svg viewBox="0 0 42 42" width="170" style="display:block;margin:0 auto">${arcs}<text x="21" y="22" text-anchor="middle" font-size="6" font-weight="700" fill="var(--ink)">${tot.toLocaleString()}</text><text x="21" y="27" text-anchor="middle" font-size="3" fill="var(--mute)">customers</text></svg>`+
 ['High','Medium','Low'].map(k=>`<div class="row" style="grid-template-columns:90px 1fr 80px"><span><i style="display:inline-block;width:10px;height:10px;border-radius:3px;background:var(--${cols[k]});margin-right:6px"></i>${k}</span><span></span><span>${D.risk[k].toLocaleString()} (${pct(D.risk[k]/tot)})</span></div>`).join(''));
// confusion matrix
const c=D.cm;
h+=card('w4','Confusion matrix (test)',`<div class="cm"><div class="h"></div><div class="h">Pred: stays</div><div class="h">Pred: churns</div>
 <div class="h">Actual: stays</div><div class="ok"><b>${c.tn}</b>correct</div><div class="bad"><b>${c.fp}</b>wasted offer</div>
 <div class="h">Actual: churns</div><div class="bad"><b>${c.fn}</b>missed churner</div><div class="ok"><b>${c.tp}</b>caught</div></div>`);
// top table
h+=card('w12','Highest-value customers to save first (flagged high-risk)',`<table><tr><th>Customer ID</th><th>Churn probability</th><th>Past net spend</th><th>Days since last order</th><th>Orders</th></tr>`+
 D.top.map(t=>`<tr><td>${t.id}</td><td><span class="pill">${pct(t.p)}</span></td><td>${money(t.value)}</td><td>${Math.round(t.recency)}</td><td>${t.orders}</td></tr>`).join('')+`</table><div class="note">Scores are produced for all customers, including those the model trained on, so use this list for prioritisation, not for evaluation.</div>`);
$('#m').innerHTML=h;
</script></body></html>"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    build_dashboard(a.config, open_browser=not a.no_open)
