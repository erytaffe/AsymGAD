"""End-to-end smoke test on a self-contained toy stream.

Builds the toy dataset, runs adaptive window construction, builds the
observation graph of the attack window, trains AsymGAD, and verifies
that the injected pivot is recovered near the top of the ranking.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _run():
    tmp = Path(tempfile.mkdtemp(prefix="asymgad_test_"))
    os.environ["ASYGAD_DATA_ROOT"] = str(tmp)

    from asymgad.data.toy import make_toy_data, toy_thresholds
    from asymgad.window import build_adaptive_windows_streaming, _annotate_attacks

    make_toy_data(data_root=tmp, seed=0)

    th = toy_thresholds()
    windows, total_events = build_adaptive_windows_streaming(
        categories=("AUTH",),
        M_min=th["M_min"], E_min=th["E_min"],
        V_min_src=th["V_min_src"], V_min_dst=th["V_min_dst"],
        verbose=False,
    )
    _annotate_attacks(windows, verbose=False)

    assert len(windows) >= 2, f"expected >=2 windows, got {len(windows)}"
    attack_windows = [w for w in windows if w.has_attack]
    assert attack_windows, "no attack window was produced"
    aw = attack_windows[0]
    assert "C5" in aw.attack_pivots, aw.attack_pivots

    from asymgad.graph_data import build_window_graph

    g = build_window_graph(aw, J=4, verbose=False)
    assert g is not None and g.N > 0 and g.E > 0
    assert g.labels.sum() == 1, "expected exactly one labeled pivot node"

    from asymgad import train_asymgad
    from asymgad.metrics import average_precision, node_ranks

    r = train_asymgad(g, epochs=20, seed=42, verbose=False)
    scores = r["scores"]
    assert scores.shape[0] == g.N
    ap = average_precision(scores, g.labels)
    ranks = node_ranks(scores, g.pivot_ids, g.pivot_names)
    print(f"toy window: N={g.N} E={g.E} AP={ap:.4f} ranks={ranks}")
    assert ap > 0.0, "AP must be positive when the pivot is found"
    assert ranks["C5"] <= 3, f"pivot C5 should rank near the top, got {ranks['C5']}"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    _run()
