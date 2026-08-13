#!/usr/bin/env python3
"""Stage-2 BH/BCT admissibility checks (issue utof/repulsive-test2#6).

Why: the delivered admissibility accepted ZERO clusters at N≤120 under
θ=0.5/leaf_size=8 (an undocumented `not is_leaf` guard excluded the only usable
clusters, and the tangent test shared θ with the spatial ratio and was binding),
while the BCT side admitted singleton×singleton blocks at EVERY θ including 0 —
so the "θ=0 ⇒ exact" acceptance gate was false at the golden's own leaf_size=2.

Gates for the pre-registered fix (issue #6 sweep: leaf admissibility, decoupled
θ_x/θ_T with default θ_T = 1.5·θ_x, cluster-size guard):
  1. θ=0 exactness holds for BH AND BCT at leaf_size 2 and 8 (zero admissible
     clusters, rel err ~machine eps vs the exact dense paths).
  2. Clustering actually fires at practical sizes: θ_x=0.5/θ_T=0.75/leaf=4 on a
     closed trefoil N=120 approximates >0 edge refs (BH) and >0 blocks (BCT).
  3. Error stays controlled and shrinks with θ: BH energy rel err ≤ 5e-2 at
     (0.5, 0.75) and smaller at (0.25, 0.5) (issue-#6 sweep: 1.7e-2 / 8.0e-3).

Run: uv run --with numpy --with scipy python oracle/check_bh_stage2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tpe_stage2_oracle as O

ALPHA, BETA, EPS = 3.0, 6.0, 1e-10


def trefoil(n):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    V = np.stack([
        np.sin(t) + 2.0 * np.sin(2.0 * t),
        np.cos(t) - 2.0 * np.cos(2.0 * t),
        -np.sin(3.0 * t),
    ], axis=1) * 0.25
    E = np.asarray([[i, (i + 1) % n] for i in range(n)], dtype=int)
    return V, E


failures = []


def check(label, ok, detail=""):
    tag = "ok  " if ok else "FAIL"
    print(f"{tag} {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def main():
    V60, E60 = trefoil(60)
    V120, E120 = trefoil(120)
    E_exact60 = O.calculate_energy(V60, E60, ALPHA, BETA, EPS)

    # 1a. BH θ=0 exactness at leaf_size 2 and 8.
    for leaf in (2, 8):
        Eb, st = O.bh_energy(V60, E60, ALPHA, BETA, EPS, theta=0.0, leaf_size=leaf)
        rel = abs(Eb - E_exact60) / max(1.0, abs(E_exact60))
        check(
            f"BH theta=0 exact [N=60 leaf={leaf}]",
            rel <= 1e-12 and st["approximated_edge_refs"] == 0,
            f"rel={rel:.2e} approx_refs={st['approximated_edge_refs']}",
        )

    # 1b. BCT θ=0 exactness at leaf_size 2 (the gate the audit falsified) and 8.
    psi = np.sin(np.arange(len(E60)) * 0.61) + 0.2
    for leaf in (2, 8):
        for kind in ("high", "low"):
            Kexact = O.exact_edge_kernel_sym(V60, E60, EPS, kind)
            Kpsi, st = O.bct_matvec_kernel(V60, E60, EPS, psi, kind, theta=0.0, leaf_size=leaf)
            rel = O.rel_norm(Kpsi, Kexact @ psi)
            check(
                f"BCT theta=0 exact [{kind} N=60 leaf={leaf}]",
                rel <= 1e-12 and st["admissible_blocks"] == 0,
                f"rel={rel:.2e} admissible_blocks={st['admissible_blocks']}",
            )

    # 2. Clustering fires at practical sizes (issue-#6 sweep settings).
    Eb, st = O.bh_energy(V120, E120, ALPHA, BETA, EPS, theta=0.5, theta_t=0.75, leaf_size=4)
    E_exact120 = O.calculate_energy(V120, E120, ALPHA, BETA, EPS)
    rel_05 = abs(Eb - E_exact120) / max(1.0, abs(E_exact120))
    check(
        "BH clustering fires [N=120 thx=0.5 thT=0.75 leaf=4]",
        st["approximated_edge_refs"] > 0,
        f"approx_refs={st['approximated_edge_refs']} direct_pairs={st['direct_pairs']} rel_err={rel_05:.2e}",
    )
    Kpsi120, stb = O.bct_matvec_kernel(
        V120, E120, EPS, np.sin(np.arange(len(E120)) * 0.61) + 0.2, "high",
        theta=0.5, theta_t=0.75, leaf_size=4,
    )
    check(
        "BCT clustering fires [N=120 thx=0.5 thT=0.75 leaf=4]",
        stb["admissible_blocks"] > 0,
        f"admissible_blocks={stb['admissible_blocks']}",
    )

    # 3. Error controlled and improving as θ tightens.
    Eb2, st2 = O.bh_energy(V120, E120, ALPHA, BETA, EPS, theta=0.25, theta_t=0.5, leaf_size=4)
    rel_025 = abs(Eb2 - E_exact120) / max(1.0, abs(E_exact120))
    check(
        "BH error controlled + monotone in theta [N=120]",
        rel_05 <= 5e-2 and rel_025 <= rel_05,
        f"rel(0.5/0.75)={rel_05:.2e} rel(0.25/0.5)={rel_025:.2e}",
    )

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1
    print("all stage-2 BH/BCT checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
