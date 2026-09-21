"""Parses docs/backtest_report.md's "Per-asset results (mean across rolls)"
table (produced by src/eval/walk_forward.py, Phase 5) into a plain dict, for
GET /api/backtest/{asset}'s side-by-side live-vs-backtested comparison.

No separate machine-readable artifact from walk_forward.py exists on disk
(it only ever wrote this markdown report -- confirmed: no .csv/.json/.pkl
sibling), so parsing the markdown table is the only way to reuse those
numbers rather than recomputing them (which would be slow -- walk-forward
retrains PPO/XGBoost from scratch per roll -- and risk silently drifting
from the numbers actually reported in docs/backtest_report.md /
docs/final_report.md).
"""
import re
from pathlib import Path

DEFAULT_REPORT_PATH = Path("docs/backtest_report.md")


def load_backtest_metrics(report_path=DEFAULT_REPORT_PATH):
    """Returns {asset: {method: {metric_name: value}}}, e.g.
    metrics["DOGE"]["ppo"]["Sharpe Ratio"].
    """
    report_path = Path(report_path)
    if not report_path.exists():
        return {}

    text = report_path.read_text()
    if "## Per-asset results" not in text or "## Basket-wide average" not in text:
        return {}
    section = text.split("## Per-asset results", 1)[1].split("## Basket-wide average", 1)[0]

    table_lines = [line for line in section.splitlines() if line.strip().startswith("|")]
    if len(table_lines) < 3:
        return {}

    header_cells = [c.strip() for c in table_lines[0].strip().strip("|").split("|")]
    metric_names = header_cells[1:]  # first column is the ('asset', 'method') key

    metrics = {}
    for line in table_lines[2:]:  # skip header + markdown separator row
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(header_cells):
            continue
        key_match = re.match(r"\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)", cells[0])
        if not key_match:
            continue
        asset, method = key_match.group(1), key_match.group(2)

        row = {}
        for name, val in zip(metric_names, cells[1:]):
            try:
                row[name] = float(val)
            except ValueError:
                row[name] = val
        metrics.setdefault(asset, {})[method] = row

    return metrics


def get_backtest_for_asset(asset, report_path=DEFAULT_REPORT_PATH):
    """Returns {method: {metric: value}} for `asset`, or None if that asset
    was never walk-forward backtested (e.g. an NFT collection -- Phase 5
    only ran on assets with both features and a real price feed, see
    src/agents/common.py's available_assets())."""
    return load_backtest_metrics(report_path).get(asset)
