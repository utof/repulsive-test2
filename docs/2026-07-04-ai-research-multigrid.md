I produced a deterministic single-file oracle and a crossing golden:

[Download `tpe_stage2_oracle.py`](sandbox:/mnt/data/tpe_stage2_oracle.py)
[Download generated `crossing_stage2_golden.json`](sandbox:/mnt/data/crossing_stage2_golden.json)

I compiled the oracle and ran it on the supplied `crossing` fixture. It emitted exact dense objects, BH energy/dE, BCT matvecs, MG solves, penalties, and a dense accepted step. On that fixture: exact energy `3.434074588325977`, dense saddle residual `~1.8e-16`, accepted `τ=1`, MG gradient saddle residual `~1.7e-16`, and BCT scalar matvec relative errors at `θ=0.5` were about `1.84e-2` for `B` and `8.12e-2` for `B⁰`.

## 0. Fixed conventions used below

I use the app’s coordinate-block flattening:

[
[x_0,\dots,x_{n-1},\ y_0,\dots,y_{n-1},\ z_0,\dots,z_{n-1}].
]

All exact dense quantities follow your ordered disjoint-pair convention, including both ((I,J)) and ((J,I)). The exact scalar energy keeps the app’s final factor (1/2); the inner-product matrix (A=B+B^0) does **not** get that energy factor.

The paper defines the tangent-point kernel as

[
k^\alpha_\beta(p,q,T)=
\frac{|T\times(p-q)|^\alpha}{|p-q|^\beta},
]

and its finite-energy lemma gives

[
s=\frac{\beta}{\alpha}-\frac1\alpha=\frac{\beta-1}{\alpha}.
]

The paper also states that (dE) has order (2s), and that the high-order Sobolev operator uses (\sigma=s-1). 

For your exponents,

[
\alpha=3,\qquad \beta=6,
]

so

[
s=\frac53,\qquad \sigma=\frac23,\qquad
2s=\frac{10}{3},\qquad
2\sigma+1=\frac73,\qquad
2\sigma+5=\frac{19}{3}.
]

The exact dense inner product is the one in your prompt: (B) is assembled from edge differences and (B^0) from edge averages. The paper’s discrete inner-product section defines (A=B+B^0), block-expands it componentwise for vector-valued functions, and gives the high- and low-order increment tables used as the dense baseline.  

Endpoints and junctions are treated purely graph-theoretically for (A): only the disjointness test (I\cap J=\emptyset) matters. An edge sharing an endpoint or a junction vertex with another edge is omitted from pair interaction; otherwise it contributes normally.

---

## 1. Barnes–Hut approximation of the exact ε-regularized energy

### 1.1 BVH data

For every edge (I=(a,b)),

[
e_I=\gamma_b-\gamma_a,\qquad
\ell_I=|e_I|,\qquad
\ell_I^\varepsilon=\ell_I+\varepsilon,
]

[
T_I=
\begin{cases}
e_I/\ell_I, & \ell_I\ge 10^{-14},\
0, & \text{otherwise},
\end{cases}
\qquad
c_I=\frac{\gamma_a+\gamma_b}{2}.
]

For a BVH node (N),

[
M_N^\varepsilon=\sum_{J\in N}\ell_J^\varepsilon,
]

[
C_N=\frac{1}{M_N^\varepsilon}\sum_{J\in N}\ell_J^\varepsilon c_J,
\qquad
\bar T_N=\frac{1}{M_N^\varepsilon}\sum_{J\in N}\ell_J^\varepsilon T_J.
]

I do **not** renormalize (\bar T_N). The paper stores length-weighted node mass, center, and average tangent, and uses the pair ((\bar T_N,C_N)) as a node tangent-point. 

The spatial radius is

[
r_x^N=\max_{J\in N}|c_J-C_N|,
]

and the tangent radius is

[
r_T^N=\max_{J\in N}|T_J-\bar T_N|.
]

### 1.2 θ-admissibility

The paper’s BH condition is stated as

[
r_x^N/|c_I-C_N|\lesssim \varepsilon,\qquad r_T^N\lesssim\varepsilon,
]

where that (\varepsilon) is an approximation threshold, not your energy regularization. 

I rename the approximation threshold to (\theta). A source node (N) is admissible for target edge (I) iff:

[
N\text{ contains only edges disjoint from }I,
]

[
d_{I,N}=|c_I-C_N|>0,
]

[
\frac{r_x^N}{d_{I,N}}\le \theta,
\qquad
r_T^N\le \theta.
]

If any edge in (N) shares a vertex with (I), the node is **not** admissible; descend until either admissible or leaf. At a leaf, evaluate exact ordered edge-pair contributions for disjoint pairs and skip non-disjoint pairs.

This is stricter than merely checking the node bounding box because it preserves your exact “neighboring and junction-sharing edges are omitted” convention.

### 1.3 How ε interacts with clusters

ε is part of your energy definition, so it enters cluster approximants, not just direct leaves.

For an accepted source node (N), the clustered ordered contribution of target edge (I=(a,b)) is

[
\widehat E_{I,N}^{\rm ord}
==========================

\frac12,
(\ell_I^\varepsilon)^{1-\alpha}
M_N^\varepsilon
\sum_{p\in{a,b}}
\frac{
\left(|e_I\times(\gamma_p-C_N)|+\varepsilon\right)^\alpha
}{
\left(|\gamma_p-C_N|+\varepsilon\right)^\beta
}.
]

The factor (1/2) appears because the exact endpoint quadrature has (\frac14\sum_{p\in I}\sum_{q\in J}), and the two source endpoints have been collapsed into one source quadrature location (C_N):

[
\frac14 \sum_{p\in I}\sum_{q\in J}
\quad\leadsto\quad
\frac14\sum_{p\in I} 2
======================

\frac12\sum_{p\in I}.
]

The full BH energy is

[
\widehat E_{\rm BH}(\gamma)
===========================

\frac12
\sum_{I\in E}
\operatorname{TraverseBH}(I,\mathrm{root}),
]

where the outer (1/2) is your app’s ordered-pair energy factor.

Direct leaf contributions are the exact app terms:

[
E_{I,J}^{\rm ord}
=================

\frac14(\ell_I^\varepsilon)^{1-\alpha}\ell_J^\varepsilon
\sum_{p\in I}\sum_{q\in J}
\frac{
\left(|e_I\times(\gamma_p-\gamma_q)|+\varepsilon\right)^\alpha
}{
\left(|\gamma_p-\gamma_q|+\varepsilon\right)^\beta
}.
]

**Important:** ε is **not** used to make a near collision admissible. Admissibility uses the unregularized center distance (d_{I,N}=|c_I-C_N|). Once a node is accepted, ε is added after the aggregate norms in the approximated kernel. This preserves the intended large weight near collision; ε only prevents exact division by zero and matches the differentiable app convention.

### 1.4 BH differential

There are two sensible differentials:

1. Differentiate the BH-approximated energy (\widehat E_{\rm BH}). This is internally consistent with using BH energy in Armijo.
2. Approximate the exact full (dE) directly, as the paper describes, by accumulating derivatives from admissible nodes and direct leaves; the paper explicitly says it approximates the full discrete differential directly rather than differentiating its BH energy approximation. 

The oracle emits option 1, using deterministic forward finite differences of (\widehat E_{\rm BH}). This is slower but unambiguous and machine-checkable.

For production analytic BH differentiation of an accepted source node, define

[
F_{I,N}
=======

\frac12M_N^\varepsilon L^{1-\alpha}
\sum_{p\in{a,b}}
R_p^\alpha D_p^{-\beta},
]

with

[
L=|e|+\varepsilon,\quad
d_p=\gamma_p-C_N,\quad
D_p=|d_p|+\varepsilon,
]

[
c_p=e\times d_p,\quad
R_p=|c_p|+\varepsilon.
]

For a perturbation,

[
dF_{I,N}
========

\frac12M_N^\varepsilon
\left[
(1-\alpha)L^{-\alpha}dL\sum_pR_p^\alpha D_p^{-\beta}
+
L^{1-\alpha}\sum_p
\left(
\alpha R_p^{\alpha-1}D_p^{-\beta}dR_p
-------------------------------------

\beta R_p^\alpha D_p^{-\beta-1}dD_p
\right)
\right],
]

where

[
dL=T_I\cdot(d\gamma_b-d\gamma_a),
]

[
dD_p=\widehat d_p\cdot d\gamma_p
\quad
(\widehat d_p=0\text{ if }|d_p|<10^{-14}),
]

[
dR_p=\widehat c_p\cdot\left((d\gamma_b-d\gamma_a)\times d_p+e\times d\gamma_p\right)
\quad
(\widehat c_p=0\text{ if }|c_p|<10^{-14}).
]

This derivative treats (C_N,M_N^\varepsilon,\bar T_N) as fixed during a target-edge contribution, exactly as the paper’s admissible-node derivative note does for the target edge. 

### 1.5 BH error model

For a fixed traversal and source node (N), with

[
\rho_x=\frac{r_x^N}{|c_I-C_N|},\qquad
\rho_T=r_T^N,
]

the 0th-order lumping error has the model

[
\left|
E_{I,N}^{\rm exact}-\widehat E_{I,N}
\right|
\lesssim
C_{I,N,\varepsilon}
(\rho_x+\rho_T),
E_{I,N}^{\rm scale},
]

where (C_{I,N,\varepsilon}) grows as accepted cluster distances approach the ε-scale because derivatives of ((|d|+\varepsilon)^{-\beta}) scale like ((|d|+\varepsilon)^{-\beta-1}). This is why admissibility must remain geometric and unsoftened: a tiny (\varepsilon) cannot justify accepting a near-collision cluster.

Practical setting:

[
\theta\in{0.25,0.5}
]

for tests; (\theta=0) disables admissible clusters and must reproduce the exact dense energy up to roundoff.

---

## 2. BCT-accelerated matvecs for both kernels of (A=B+B^0)

The paper’s BCT represents edge-edge kernel matrices

[
K_{IJ}=k(p_I,p_J)\ell_I\ell_J
]

by low-rank blocks; for admissible blocks it uses the rank-1 form

[
\widehat K_{TS}
===============

m_T,k(\bar p_T,\bar p_S),m_S^\top,
]

so a block update is a dot product with source masses followed by a scaled target mass vector. 

Again, I use (\ell^\varepsilon) masses and ((|\cdot|+\varepsilon)) denominators in the approximated kernels.

### 2.1 Operators (E) and (D)

Let

[
(Eu)*I=\frac{u*{i_1}+u_{i_2}}2
]

and

[
(Du)_I
======

\frac{u_{i_2}-u_{i_1}}{\ell_I^\varepsilon}T_I
\in\mathbb R^3.
]

For vector-valued data, apply the scalar construction independently per coordinate block.

### 2.2 High-order kernel

The exact dense high-order form is

[
u^\top Bv
=========

\sum_{I,J,\ I\cap J=\emptyset}
w_{IJ}
\langle D_Iu-D_Ju,\ D_Iv-D_Jv\rangle.
]

Because the exact ordered pair list includes both ((I,J)) and ((J,I)), define the symmetric edge kernel

[
K^H_{IJ}=
\begin{cases}
w_{IJ}+w_{JI},& I\cap J=\emptyset,\ I\ne J,\
0,& \text{otherwise}.
\end{cases}
]

Here (w_{IJ}=w_{JI}) for the high-order denominator-only weight, so (K^H_{IJ}=2w_{IJ}). Then

[
B u
===

D^\top
\left[
\operatorname{diag}(K^H\mathbf 1)-K^H
\right]
Du.
]

The BCT midpoint approximant for an admissible block uses

[
k_H(\bar p_T,\bar p_S)
======================

\frac{2}{(|C_T-C_S|+\varepsilon)^{7/3}}.
]

The factor (2) is the ordered-pair symmetrization.

Direct BCT leaves use the exact endpoint quadrature for (w_{IJ}+w_{JI}), not midpoint quadrature.

### 2.3 Low-order kernel

The exact dense low-order form is

[
u^\top B^0v
===========

\sum_{I,J,\ I\cap J=\emptyset}
w^0_{IJ}(u_I-u_J)(v_I-v_J).
]

Define

[
K^0_{IJ}=
\begin{cases}
w^0_{IJ}+w^0_{JI},& I\cap J=\emptyset,\ I\ne J,\
0,& \text{otherwise}.
\end{cases}
]

Then

[
B^0u
====

E^\top
\left[
\operatorname{diag}(K^0\mathbf1)-K^0
\right]
Eu.
]

This matches the paper’s decomposition of the low-order matrix into (E^\top(\operatorname{diag}(K\mathbf1)-K)E), with (E) the vertex-to-edge averaging operator. The paper gives the low-order kernel as the sum of two oriented midpoint tangent-point kernels, and says the high-order part is the same decomposition with (E) replaced by (D) and the kernel replaced by (k^0_{2\sigma+1}). 

The BCT admissible-block midpoint approximant is

[
k_0(\bar p_T,\bar p_S)
======================

\frac{
(|\bar T_T\times(C_T-C_S)|+\varepsilon)^2
+
(|\bar T_S\times(C_T-C_S)|+\varepsilon)^2
}{
(|C_T-C_S|+\varepsilon)^{4+7/3}
}.
]

Since (4+7/3=19/3),

[
k_0(\bar p_T,\bar p_S)
======================

\frac{
(|\bar T_T\times d|+\varepsilon)^2
+
(|\bar T_S\times d|+\varepsilon)^2
}{
(|d|+\varepsilon)^{19/3}
},
\qquad d=C_T-C_S.
]

Direct BCT leaves use exact endpoint (w^0_{IJ}+w^0_{JI}).

### 2.4 BCT admissibility and disjointness

A BCT block ((T,S)) is admissible iff:

[
\frac{\max(r_x^T,r_x^S)}{|C_T-C_S|}\le\theta,
\qquad
\max(r_T^T,r_T^S)\le\theta,
]

and every target edge in (T) is disjoint from every source edge in (S). The paper’s BCT condition is the same geometric/tangent coherence condition, again using its symbol (\varepsilon) for approximation error, not your energy ε. 

If the all-disjoint test fails, descend. At a leaf, sum exact disjoint edge pairs and skip non-disjoint pairs.

---

## 3. Geometric multigrid for the two saddle solves

The paper warns that multigrid on the whole saddle matrix is not the intended method; instead it projects to the nullspace of the constraint matrix and solves a projected system. 

### 3.1 Constraint stack

Use one ordered stack:

[
C=
\begin{bmatrix}
C_{\rm bar}\
C_{\rm totalLength}\ \text{or}\ C_{\rm edgeLengths}\
C_{\rm point}
\end{bmatrix}.
]

Row counts:

[
k=
3
+
\begin{cases}
0,&\text{no length block},\
1,&\text{totalLength},\
|E|,&\text{edgeLengths},
\end{cases}
+
3,n_{\rm pins}.
]

Do **not** enable `totalLength` and `edgeLengths` simultaneously by default. That XOR avoids redundant length constraints.

For projection of the saddle system, define

[
C^\dagger=C^\top(CC^\top)^\dagger,
\qquad
\Pi=I-C^\dagger C.
]

If (C) is full row rank, this is the usual

[
C^\dagger=C^\top(CC^\top)^{-1}.
]

Use SVD or rank-revealing QR for (CC^\top) in tests so rank deficiencies fail cleanly.

### 3.2 Gradient solve

Dense saddle form:

[
\begin{bmatrix}
\bar A & C^\top\
C&0
\end{bmatrix}
\begin{bmatrix}
\tilde g\ \lambda
\end{bmatrix}
=============

\begin{bmatrix}
dE\0
\end{bmatrix}.
]

Projected MG form:

[
\Pi^\top \bar A\Pi y=\Pi^\top dE,
\qquad
\tilde g=\Pi y.
]

Recover multipliers for diagnostics by solving

[
CC^\top\lambda=C(dE-\bar A\tilde g).
]

### 3.3 Constraint-projection solve

For a candidate (\gamma^q), let (b=-\Phi(\gamma^q)). The dense saddle solve is

[
\begin{bmatrix}
\bar A & C^\top\
C&0
\end{bmatrix}
\begin{bmatrix}
x\ \mu
\end{bmatrix}
=============

\begin{bmatrix}
0\b
\end{bmatrix}.
]

Use a particular solution

[
z=C^\dagger b,
\qquad Cz=b.
]

Then solve

[
\Pi^\top \bar A\Pi y=\Pi^\top \bar A z,
\qquad
x=z-\Pi y.
]

This is the same transformation described in the paper for constraint projection. 

### 3.4 Coarsening for general curve networks

The paper’s base rule is: mark alternating vertices black/white, force endpoints and junctures black, remove whites, preserve black values, and interpolate a white vertex by averaging its two neighboring black vertices. 

I make that rule deterministic for arbitrary graph networks:

1. Compute graph degree from the edge list.
2. Force black:
   [
   \deg(v)=0,\quad \deg(v)=1,\quad \deg(v)\ge3,\quad v\in\text{point pins}.
   ]
3. Treat every maximal degree-2 chain between black anchors independently.
4. Along each chain, mark alternating vertices black/white.
5. Never interpolate across a junction. A white vertex on a branch averages only the two black vertices on that same branch.
6. For a closed all-degree-2 component, pick the minimum original vertex index as the first black vertex and alternate. If an odd cycle would leave the last interpolation ambiguous, force one additional black vertex.
7. Multiple components are coarsened independently.
8. Coarse edges represent paths between consecutive black vertices along each chain.
9. Coarse edge-length targets for `edgeLengths` are path-sums of fine target lengths. The total-length target is unchanged. Barycenter target (x_0) is unchanged. Point pins remain black and have identical targets on every level.

Prolongation (P_{\ell}^{\ell-1}) is scalar (n_f\times n_c); vector prolongation is block diagonal:

[
\bar P=\operatorname{diag}(P,P,P).
]

For black fine vertex (i) corresponding to coarse vertex (I),

[
P_{iI}=1.
]

For a white fine vertex (i) between coarse neighbors (I,J),

[
P_{iI}=P_{iJ}=\frac12.
]

I keep the paper’s unweighted average by default; length-weighted interpolation is a tunable variant but not the oracle default.

### 3.5 Smoother, cycle, and coarse solve

Default scheme:

[
\nu_{\rm pre}=2,\qquad \nu_{\rm post}=2,
]

with projected conjugate-gradient smoothing on

[
M_\ell=\Pi_\ell^\top \bar A_\ell \Pi_\ell.
]

The paper says it uses a standard CG smoother and typically needs six or fewer V-cycles to reach residual (10^{-3}); it also says driving the residual lower has diminishing returns when the BCT itself is approximate. 

Use a V-cycle by default. Switch to a W-cycle only if the projected residual reduction factor over two consecutive V-cycles is worse than (0.7).

Coarse solve threshold:

[
n_\ell\le64
]

or no removable white vertices. Coarse solve is dense saddle or dense projected SPD. In the oracle, I allow a final dense cleanup to certify against the exact saddle residual.

### 3.6 Stopping criteria

There are two regimes.

For dense or oracle-backed MG:

[
\frac{|Kz-rhs|_2}{\max(1,|rhs|_2)}\le10^{-10}.
]

This is compatible with your existing self-certifying residual gate.

For production BCT+MG without exact dense (K), require:

[
\frac{|\widehat Kz-rhs|_2}{\max(1,|rhs|_2)}\le10^{-10}
]

against the approximate operator, plus one of:

[
\frac{|K_{\rm exact}z-rhs|_2}{\max(1,|rhs|_2)}
\le
\max(10^{-10},20\delta_A)
]

on calibration fixtures where dense (K_{\rm exact}) is available, or a measured BCT matvec relative error bound (\delta_A(\theta)) no larger than the solver residual target required by the application.

This replacement is necessary because a BCT operator with (\theta>0) has a deterministic consistency error; iterating MG below the operator error floor does not improve the exact residual.

For actual descent safety, always additionally require:

[
dE^\top \tilde g>0
]

and accepted Armijo decrease after projection.

### 3.7 Projection tolerance note

For solver residuals, keep (10^{-10}). For nonlinear projection tolerances, the uploaded oracle README distinguishes the frozen Stage-1 `1e-10` contract from the later reference-code projection tolerance `1e-4`; it says the `1e-4` value comes from the authors’ reference implementation, while the per-block scaled rule is your own. 

For this Stage-2 oracle, I defaulted nonlinear projection to `1e-10` to preserve golden comparability. In production, using the later `1e-4` projection tolerance is defensible, but it should be recorded as a policy choice rather than a paper formula.

---

## 4. Time stepping constants

The paper’s simpler application strategy is: normalize the gradient, start line search at (\tau=1), backtrack until Armijo holds and constraint projection succeeds, and stop when the (L^2) norm of the fractional Sobolev gradient is below (10^{-4}). 

Concrete app-compatible constants:

[
c_1=10^{-4},\qquad \rho=\frac12,\qquad \tau_{\min}=10^{-12}.
]

Let

[
|g|*{L^2_h}^2=\sum*{I\in E}\ell_I\frac{|g_{i_1}|^2+|g_{i_2}|^2}{2}.
]

Normalize

[
p=\tilde g/\max(|\tilde g|_{L^2_h},10^{-300}).
]

For a candidate

[
\gamma_{\rm cand}=\gamma-\tau p,
]

project to (\gamma_{\rm proj}). Accept if projection succeeds and

[
E(\gamma_{\rm proj})
\le
E(\gamma)-c_1\tau, dE^\top p.
]

Terminate if

[
|\tilde g|_{L^2_h}<10^{-4}.
]

If (\tau<\tau_{\min}), reject the step and report `line_search_failed`, not convergence.

---

## 5. Oracle contents

The delivered file:

[Download `tpe_stage2_oracle.py`](sandbox:/mnt/data/tpe_stage2_oracle.py)

It is single-file, deterministic, and uses only `numpy` and `scipy`. It reads:

```json
{
  "vertices": [[x,y,z], ...],
  "edges": [[i,j], ...],
  "alpha": 3,
  "beta": 6,
  "epsilon": 1e-10,
  "theta": 0.5,
  "leaf_size": 16
}
```

and writes JSON containing:

* exact energy and FD (dE),
* exact (B,B^0,A,\bar A),
* barycenter constraint (\Phi,C),
* dense constrained Sobolev gradient and accepted step,
* BH energy and BH FD (dE) for `theta/2` and `theta`,
* BCT high/low kernel matvecs and scalar (B,B^0) matvecs,
* MG gradient and projection solves, compared to dense,
* optional penalty differentials,
* property summaries and timings.

The earlier Stage-1 oracle harness is intended to compute energy, FD (dE), (B,B^0,A,\bar A), barycenter (\Phi/C), constrained gradient, and one accepted line-search step, with correctness established by diffing against the TS implementation.  The handoff also says the oracle must be single-file, deterministic, numpy/scipy only, and directly diffable against the TS implementation. 

---

## 6. Machine-checkable property checklist

### Exact dense baseline

Use these on every fixture:

1. Symmetry:
   [
   |B-B^\top|*\infty\le10^{-14}\max(1,|B|*\infty)
   ]
   and same for (B^0,A).

2. PSD:
   [
   \lambda_{\min}(A)\ge -10^{-9}|A|_2.
   ]

3. Constant nullspace:
   [
   |A\mathbf1|_2\le10^{-10}\max(1,|A|_2).
   ]

4. Quadratic-form identity:
   direct double-sum (u^\top Bv) and assembled (u^\top Bv) agree to (10^{-12}) relative; same for (B^0).

5. Dense saddle residual:
   [
   \frac{|Kz-rhs|_2}{\max(1,|rhs|_2)}\le10^{-10}.
   ]

6. Constraint tangency:
   [
   |C\tilde g|_2\le10^{-10}\max(1,|\tilde g|_2).
   ]

7. Descent:
   [
   dE^\top\tilde g>0.
   ]

8. Accepted-step energy:
   [
   E_{\rm new}<E_{\rm old}
   ]
   unless already converged.

9. Orientation invariance: reversing an edge orientation changes neither energy nor (A) beyond (10^{-12}) relative.

10. Translation/rotation invariance: energy and (A) invariant under rigid motions to (10^{-12}) relative.

11. Scale laws with (\varepsilon=0):
    [
    E(c\gamma)=c^{-1}E(\gamma),
    \qquad
    A(c\gamma)=c^{-7/3}A(\gamma).
    ]
    With (\varepsilon=10^{-10}), test this only when all lengths and distances are (\gg\varepsilon), with tolerance (10^{-8})–(10^{-6}) depending on scale.

The uploaded checklist already includes symmetry, PSD, constant nullspace, quadratic-form identity, saddle residuals, descent, energy decrease, and barycenter preservation; it also states TS-vs-oracle matrices/vectors should diff at (\le10^{-8}) relative. 

### BH error vs θ

For each nondegenerate fixture:

* `θ=0` must reproduce exact energy:
  [
  |E_{\rm BH}-E|/\max(1,|E|)\le10^{-14}.
  ]

* `θ=0.25`:
  [
  |E_{\rm BH}-E|/\max(1,|E|)\le5\times10^{-3},
  ]
  [
  |dE_{\rm BH}-dE_{\rm exactFD}|/\max(1,|dE_{\rm exactFD}|)
  \le5\times10^{-2}.
  ]

* `θ=0.5`:
  [
  |E_{\rm BH}-E|/\max(1,|E|)
  \le5\times10^{-2},
  ]
  [
  |dE_{\rm BH}-dE_{\rm exactFD}|/\max(1,|dE_{\rm exactFD}|)
  \le2\times10^{-1}.
  ]

The derivative tolerance is looser because the oracle’s BH differential is forward FD with (h=10^{-6}), so it includes FD truncation and traversal discontinuity effects.

### BCT matvecs

For deterministic probe vectors:

* `θ=0`: exact direct BCT matvec equals dense matvec to (10^{-12}) relative.
* `θ=0.25`:
  [
  |K_{\rm BCT}\psi-K\psi|/\max(1,|K\psi|)
  \le10^{-2}
  ]
  for both high and low kernels on standard fixtures.
* `θ=0.5`:
  [
  \le10^{-1}.
  ]
* For scalar (B) and (B^0) matvecs:
  [
  |B_{\rm BCT}u-Bu|/\max(1,|Bu|)
  \le3\times10^{-2}
  ]
  at `θ=0.25`, and (\le1.5\times10^{-1}) at `θ=0.5`.

These are acceptance tests for the 0th-order approximation, not mathematical sharp bounds.

### MG vs dense

On fixtures with (n\le300), compare MG against dense saddle solves:

[
|x_{\rm MG}-x_{\rm dense}|*2/\max(1,|x*{\rm dense}|_2)
\le10^{-8}.
]

Self-certified residual:

[
|Kz-rhs|_2/\max(1,|rhs|_2)\le10^{-10}
]

when using dense cleanup. Without cleanup, require:

[
\le10^{-8}
]

for pure MG or

[
\le\max(10^{-10},20\delta_A)
]

for BCT-backed MG.

Iteration gates:

* Dense-cleanup oracle: no more than 20 V-cycles before cleanup.
* Pure MG target: no more than 8 V-cycles to (10^{-6}), no more than 20 to (10^{-8}), on bundled fixtures.
* Paper-style loose preconditioning: no more than 6 V-cycles to (10^{-3}), matching the paper’s statement. 

### Near-linear scaling demonstration

Generate open-chain or loop refinements with (N=64,128,256,512,1024), fixed geometry, fixed (\theta=0.5), fixed leaf size.

Measure BCT matvec wall time and number of BCT leaves. Accept if the log-log slope for BCT matvec time is

[
\le1.35
]

over the last four resolutions, and memory/leaves slope is

[
\le1.35.
]

Dense matvec/storage should show slope near 2, confirming the contrast. The paper reports near-linear scaling for the full accelerated scheme and super-quadratic behavior for dense solves. 

---

## 7. Optional penalty differentials

### 7.1 Total length

[
P_L(\gamma)=\sum_{I=(a,b)}\ell_I.
]

For each edge:

[
\nabla_{\gamma_a}P_L\mathrel{+}= -T_I,
\qquad
\nabla_{\gamma_b}P_L\mathrel{+}= +T_I,
]

with (T_I=0) if (\ell_I<10^{-14}).

### 7.2 Length-difference penalty

For degree-2 vertices only,

[
P_{\rm diff}=\sum_{v\in V_{\rm int}}(\ell_{I_v}-\ell_{J_v})^2.
]

Choose (I_v,J_v) deterministically as the two incident edges in increasing edge-index order unless your graph stores curve-order adjacency. Let

[
\Delta_v=\ell_{I_v}-\ell_{J_v}.
]

Then

[
dP_v=2\Delta_v,d\ell_{I_v}-2\Delta_v,d\ell_{J_v}.
]

For an edge (I=(a,b)),

[
d\ell_I=T_I\cdot(d\gamma_b-d\gamma_a),
]

so add (2\Delta_v(-T_I,+T_I)) to (I_v) and (-2\Delta_v(-T_J,+T_J)) to (J_v).

Junctions of degree (\ge3) and endpoints are excluded from (V_{\rm int}).

### 7.3 Field potential

[
P_X(\gamma)=\sum_{I=(a,b)} \ell_I |T_I\times X(c_I)|^2,
\qquad c_I=\frac{\gamma_a+\gamma_b}{2}.
]

Assume (X:\mathbb R^3\to\mathbb R^3) is differentiable and provide (J_X(c)=\nabla X(c)). If (J_X) is unavailable, use FD for this penalty’s differential.

Let

[
X_I=X(c_I),\qquad s_I=T_I\cdot X_I,
]

[
f_I=|T_I\times X_I|^2=|X_I|^2-s_I^2.
]

Define

[
g_T=2(|X_I|^2T_I-s_I X_I),
]

[
g_X=2(X_I-s_I T_I),
]

[
P_T=I-T_IT_I^\top.
]

Then the edge contribution has

[
h_e=f_I T_I+P_Tg_T,
]

[
h_c=\frac{\ell_I}{2}J_X(c_I)^\top g_X.
]

The vertex gradients are

[
\nabla_{\gamma_a}P_X\mathrel{+}= -h_e+h_c,
\qquad
\nabla_{\gamma_b}P_X\mathrel{+}= +h_e+h_c.
]

For constant unit fields, (J_X=0).

FD check: central differences with (h=10^{-6}) should agree to (10^{-6}) relative for total length and length-difference, and (10^{-5}) relative for field potential with a smooth analytic (X,J_X).

---

## 8. Ledger of paper-unspecified choices

These are implementation choices I made, not paper facts:

1. **Approximation threshold symbol:** the paper uses (\varepsilon) for BH/BCT approximation error; I renamed it (\theta) to avoid collision with your energy ε.
2. **BH cluster formula for your ε-regularized endpoint-quadrature energy:** the paper gives a midpoint tangent-point lumping formula, not your exact endpoint ε formula. I chose endpoint target quadrature with source endpoints collapsed to (C_N).
3. **Cluster masses:** I used (\ell^\varepsilon), not raw (\ell), because your exact energy uses (\ell^\varepsilon) in weights.
4. **Cluster centers/tangents:** I used (\ell^\varepsilon)-weighted centers and average tangents, without normalizing the average tangent.
5. **Admissibility distance:** I used unregularized center distances. ε never makes a near-collision admissible.
6. **Cluster ε placement:** I add ε after aggregate norms (|\gamma_p-C_N|) and (|e_I\times(\gamma_p-C_N)|), matching your “epsilon after norm” rule.
7. **BH disjointness:** admissible nodes must be entirely disjoint from the target edge. Mixed nodes descend.
8. **BH differential oracle:** I emit FD of the BH approximated energy. The paper’s direct differential approximation is recorded, but not used as the oracle’s golden differential.
9. **BCT direct leaves:** direct leaves use exact endpoint quadrature kernels, not midpoint kernels.
10. **BCT rank:** I use only the paper’s 0th-order rank-1 approximation.
11. **BCT symmetric kernels:** I explicitly symmetrize ordered high and low kernels as (K_{IJ}=w_{IJ}+w_{JI}).
12. **MG coarsening on closed degree-2 components:** I choose the minimum original vertex index as the first black vertex.
13. **Junction handling:** junction vertices are forced black and interpolation never crosses them.
14. **Point pins:** pinned vertices are forced black on all levels.
15. **Coarse edge-length targets:** for `edgeLengths`, coarse targets are path-sums of fine targets.
16. **Constraint policy:** `totalLength` XOR `edgeLengths` by default.
17. **Smoother:** projected CG, two pre- and two post-smoothing iterations.
18. **Cycle:** V-cycle default, W-cycle only on stagnation.
19. **Coarsest size:** dense solve once (n\le64) or no removable vertices remain.
20. **Residual policy for approximate operators:** exact (10^{-10}) residual only when dense exact (K) is available; otherwise use approximate residual plus a calibrated BCT error floor.
21. **Line-search constants:** (c_1=10^{-4}), (\rho=1/2), (\tau_{\min}=10^{-12}), and the mass-lumped (L^2_h) norm are app/oracle choices, not paper constants. The uploaded audit explicitly flags these constants as tunable choices rather than paper constants. 
22. **Projection tolerance:** this answer’s Stage-2 oracle uses `1e-10` for golden comparability; the later production constraint README supports `1e-4` as a reference-implementation tolerance. 
23. **Field potential analytic differential:** requires (J_X=\nabla X). If your API only supplies (X), (J_X) is missing and this penalty should be FD-checked or differentiated by AD.

No core symbol needed for (A), the two saddle systems, BH/BCT structure, or MG projection was missing from the excerpts; what was missing was mainly constants and the exact ε-regularized clustered discretization.
