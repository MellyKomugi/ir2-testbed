"""Builds RESULTS.md from baselines/scores.json and pairs/*/results.json. Runs by itself on every push to main."""
import json
from pathlib import Path
import numpy as np
import testbed

NAMES = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna"}
HEAD = "| Method | nDCG@10 | Recall@100 | Compared with | Δ nDCG@10 [95 % CI] | p |\n|---|:-:|:-:|---|:-:|:-:|"


def row(pair, r, name):
    c = r["ndcg10_vs_baseline"]
    delta = f"{c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
    if c["p"] < 0.05 and c["ci95"][0] * c["ci95"][1] > 0:      # bold only what survives both tests
        delta = f"**{delta}**"
    p = "< 0.001" if c["p"] < 0.001 else f"{c['p']:.3f}"
    return f"| {pair['method']} ([{pair['pair']}]({name})) | {r['ndcg10']:.3f} | {r['recall100']:.3f} | {testbed.LABELS[r['baseline']['name']]} | {delta} | {p} |"


def main():
    base = testbed.stored()
    pairs = []
    for path in sorted(testbed.ROOT.glob("pairs/[0-9]*/results.json")):
        pairs.append((json.load(open(path)), f"pairs/{path.parent.name}/"))
    lines = ["# Shared results table", "",
             "Built by `build_table.py` from `baselines/scores.json` and `pairs/*/results.json`: do not edit by hand.",
             "A difference is in bold only when its p-value is below 0.05 and its confidence interval excludes zero.",
             "Pair 00 is the instructor's worked example from the model presentation.", ""]
    for name in testbed.DATASETS:
        lines += [f"## {NAMES[name]}", "", HEAD]
        for s in testbed.SYSTEMS:
            if s in base.get(name, {}):
                nd, rc = np.mean(base[name][s]["ndcg10"]), np.mean(base[name][s]["recall100"])
                lines.append(f"| *{testbed.LABELS[s]}* | {nd:.3f} | {rc:.3f} | | | |")
        for pair, folder in pairs:
            if name in pair["datasets"]:
                lines.append(row(pair, pair["datasets"][name], folder))
        lines.append("")
    (testbed.ROOT / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
