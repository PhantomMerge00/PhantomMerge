"""Appendix table: trajectory PM% and P(PM|TC)% with 95% CI from frozen RQ1 bootstrap."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ccer.paths import ARTIFACTS


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}\\%"


def _pct_md(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _ci_md(lo: float, hi: float) -> str:
    return f"[{_pct_md(lo)}, {_pct_md(hi)}]"


def main() -> int:
    src = ARTIFACTS / "rq1" / "four_domain_gold_stats.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    order = ("shopping", "telecom", "airline", "retail")
    rows: list[dict] = []
    for dom in order:
        d = payload["domains"][dom]
        boot = d["bootstrap_ci"]
        pm = boot["trajectory_pm"]
        tc = boot["task_correct_pm"]
        rows.append(
            {
                "domain": dom,
                "n": d["n_trajectories"],
                "pm_rate": pm["rate"],
                "pm_ci_low": pm["ci_low"],
                "pm_ci_high": pm["ci_high"],
                "tc_n": d["task_correct_n"],
                "tc_pm_rate": tc["rate"],
                "tc_ci_low": tc["ci_low"],
                "tc_ci_high": tc["ci_high"],
                "n_boot": boot.get("n_boot", 5000),
                "method": pm.get("method", "cluster_bootstrap_trajectory"),
            }
        )

    out_json = ARTIFACTS / "rq1" / "appendix_pm_tc_bootstrap_ci.json"
    out_json.write_text(
        json.dumps(
            {
                "schema": "appendix_pm_tc_bootstrap_ci_v1",
                "source": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
                "n_boot": 5000,
                "method": "cluster_bootstrap_trajectory",
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# 附录：轨迹级 PM 与 P(PM | task-correct)（95% CI）",
        "",
        "数值来源：`rq1/four_domain_gold_stats.json` → `bootstrap_ci`（轨迹聚类 bootstrap，**B=5000**，`n_boot_effective=5000`）。",
        "不重算；仅摘录 `trajectory_pm` 与 `task_correct_pm` 两行指标。",
        "",
        "| Domain | N | PM (traj%) [95% CI] | TC N | P(PM \\| TC) [95% CI] |",
        "|--------|--:|---------------------|-----:|------------------------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['domain']} | {r['n']} | {_pct_md(r['pm_rate'])} {_ci_md(r['pm_ci_low'], r['pm_ci_high'])} | "
            f"{r['tc_n']} | {_pct_md(r['tc_pm_rate'])} {_ci_md(r['tc_ci_low'], r['tc_ci_high'])} |"
        )
    md_lines.append("")
    md_lines.append(
        "**定义**：**PM** = 轨迹含 ≥1 PM-core claim 的比例；**P(PM|TC)** = 在域内 task-correct 协议判定为 TC 的轨迹子集上 PM 比例（Shopping：top-1 PID；τ³：见 RQ1 脚注）。"
    )
    md_path = ARTIFACTS / "rq1" / "APPENDIX_PM_TC_BOOTSTRAP_CI.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        "% Auto-generated from four_domain_gold_stats.json (B=5000 trajectory-cluster bootstrap)",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Trajectory-level PM prevalence and $P(\\text{PM}\\mid\\text{TC})$ with 95\\% CIs "
        "(cluster bootstrap, $B{=}5000$). Source: RQ1 gold export; CI fields \\texttt{trajectory\\_pm} "
        "and \\texttt{task\\_correct\\_pm} in \\texttt{four\\_domain\\_gold\\_stats.json}.}",
        "\\label{tab:appendix-pm-tc-ci}",
        "\\begin{tabular}{l r c r c}",
        "\\toprule",
        "Domain & $N$ & PM (traj.) [95\\% CI] & TC $N$ & $P(\\text{PM}\\mid\\text{TC})$ [95\\% CI] \\\\",
        "\\midrule",
    ]
    for r in rows:
        pm_cell = (
            f"{_pct(r['pm_rate'])} "
            f"[{_pct(r['pm_ci_low'])}, {_pct(r['pm_ci_high'])}]"
        )
        tc_cell = (
            f"{_pct(r['tc_pm_rate'])} "
            f"[{_pct(r['tc_ci_low'])}, {_pct(r['tc_ci_high'])}]"
        )
        tex_lines.append(
            f"{r['domain'].capitalize()} & {r['n']} & {pm_cell} & {r['tc_n']} & {tc_cell} \\\\"
        )
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    tex_path = ARTIFACTS / "rq1" / "tables" / "tab_appendix_pm_pm_given_tc_ci.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
