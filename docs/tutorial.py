import re
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def normalize_latex(md_text: str) -> str:
    # display math first
    md_text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", md_text, flags=re.S)
    # inline math
    md_text = re.sub(r"\\\((.*?)\\\)", r"$\1$", md_text, flags=re.S)
    return md_text


def md(s):
    cells.append(nbf.v4.new_markdown_cell(normalize_latex(s)))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md(r"""
# Analytical Gradient of the Discrete Tangent‑Point Energy (Tutorial + Coding Guide)

This is a tutorial/lecture-style walkthrough of how to derive and implement the **analytical gradient** of the **discrete tangent‑point energy** used for repulsive curve optimization.

**Audience:** undergrad / early grad (comfortable with multivariable calculus and vectors).  
**Goal:** after reading, you should be able to:
1. derive the gradient yourself (without “black box” steps),
2. implement it correctly,
3. debug sign mistakes confidently via finite-difference checks.

---

## References (for verification)
- **Yu, Schumacher, Crane (2020)**, *Repulsive Curves* (arXiv:2006.07859). Definition of discrete tangent‑point energy and kernel.
- Any standard vector calculus source for cross/dot/triple-product identities (e.g. **Arfken & Weber**, *Mathematical Methods for Physicists*; or **Wikipedia** pages for “vector triple product” and “scalar triple product”).
- Derivative rules for norms and powers: the “**Matrix Cookbook**” (Petersen & Pedersen) is a good compact reference.

This notebook focuses on the “why” of each step, and then verifies correctness numerically.
""")

md(r"""
## 0. Prerequisites (what you must already know)

You should be comfortable with:

### Vectors in 3D
- Dot product: \(a\cdot b\)
- Cross product: \(a\times b\)
- Norm: \(\|a\| = \sqrt{a\cdot a}\)

### Multivariable calculus
- Gradient of a scalar function: \(\nabla_x f(x)\)
- Chain rule for compositions: \(f(g(x))\)

### A common “physics style” differential trick
Instead of memorizing Jacobians, we’ll often write small variations:
- Let \(x \mapsto x + \delta x\).
- Compute the resulting \(\delta f\).
- Identify the gradient from \(\delta f = \nabla_x f \cdot \delta x\).

This is one of the cleanest ways to differentiate expressions involving cross products.

---
""")

md(r"""
## 1. The discrete tangent‑point energy (what we’re differentiating)

We have a polygonal curve (or general graph) in \(\mathbb{R}^3\):
- vertices \( \gamma_i \in \mathbb{R}^3\)
- edges \(I=(i_1,i_2)\)

### Edge geometry
For an oriented edge \(I=(i_1,i_2)\), define:
- edge vector: \( e_I = \gamma_{i_2}-\gamma_{i_1}\)
- length: \( \ell_I = \|e_I\|\)
- unit tangent: \( T_I = \frac{e_I}{\ell_I}\)

### Tangent‑point kernel
For points \(p,q\) and a tangent direction \(T\),
\[
k_{\beta}^{\alpha}(p,q,T)=\frac{\|T\times(p-q)\|^\alpha}{\|p-q\|^\beta},
\qquad \alpha>1,\ \beta\ge \alpha+2.
\]

### Discrete energy (edge–edge form)
For two disjoint edges \(I,J\) (no shared vertices), the discrete energy sums 4 point–point samples (each endpoint of \(I\) against each endpoint of \(J\)):
\[
\hat{\mathcal{E}}_{\beta}^{\alpha}(\gamma)
= \sum_{I\cap J=\emptyset}\frac14\,\ell_I\,\ell_J\sum_{i\in I}\sum_{j\in J}
k_{\beta}^{\alpha}(\gamma_i,\gamma_j,T_I).
\]

That “\(\frac14\)” is just the average over the 4 endpoint pairs.

---

## 1.1 Key simplification: eliminate the unit tangent

Because \(T_I=e_I/\ell_I\), we have
\[
T_I\times d = \frac{e_I\times d}{\ell_I},
\qquad d=(p-q).
\]

So
\[
\|T_I\times d\|^\alpha = \frac{\|e_I\times d\|^\alpha}{\ell_I^\alpha}.
\]

Plugging into the discrete energy gives, for one disjoint pair \((I,J)\),
\[
E_{IJ}=\frac14\,\ell_J\,\ell_I^{\,1-\alpha}\sum_{i\in I}\sum_{j\in J}
\underbrace{\frac{\|e_I\times d_{ij}\|^\alpha}{\|d_{ij}\|^\beta}}_{=:K_{ij}},
\qquad d_{ij}=\gamma_i-\gamma_j.
\]

So everything reduces to differentiating
\[
K(e,d)=\frac{\|e\times d\|^\alpha}{\|d\|^\beta},
\]
plus the prefactors \(\ell_J\) and \(\ell_I^{1-\alpha}\).

This is the main computational win: **you avoid differentiating a normalization inside a cross product**.

---
""")

md(r"""
## 2. Identities you will use (and what they mean)

### 2.1 Linearization of the cross product
For small variations \(\delta a,\delta b\),
\[
\delta(a\times b)=\delta a\times b + a\times \delta b.
\]
This is just bilinearity of \(\times\).

### 2.2 Scalar triple product (reordering dot/cross)
\[
u\cdot (v\times w) = v\cdot (w\times u) = w\cdot (u\times v).
\]
This identity is extremely useful for turning “dot of cross” into “dot of something else”.

### 2.3 Vector triple product (BAC–CAB)
\[
a\times (b\times c)= b(a\cdot c)-c(a\cdot b).
\]

Two special cases we’ll need constantly:
\[
d\times(e\times d)=\|d\|^2 e - (d\cdot e)d,
\]
\[
(e\times d)\times e=\|e\|^2 d-(d\cdot e)e.
\]

### 2.4 Norm derivatives
For \(r(x)=\|x\|\) (assuming \(x\neq 0\)),
\[
\delta \|x\| = \frac{x}{\|x\|}\cdot \delta x,
\qquad\Rightarrow\qquad \nabla_x \|x\| = \frac{x}{\|x\|}.
\]
And for powers,
\[
\nabla_x \|x\|^p = p\|x\|^{p-2}\,x.
\]

### 2.5 A sign pitfall (optional, but worth knowing)
Many implementations use the skew matrix \([a]_\times\) such that \([a]_\times b = a\times b\).
A correct and frequently-misused row identity is:
\[
v^T [a]_\times = (v\times a)^T
\]
(not \((a\times v)^T\)). Getting this wrong flips signs in the gradient.

We’ll avoid this pitfall by differentiating with differentials (Section 3).
""")

md(r"""
## 3. Derivative of the core kernel \(K(e,d)\)

Define:
- \(c(e,d) := e\times d\)
- \(r_c := \|c\|\)
- \(r_d := \|d\|\)
- \(K(e,d) = r_c^\alpha\, r_d^{-\beta}\)

We compute gradients \(\nabla_e K\) and \(\nabla_d K\).

### 3.1 First: derivative of \(r_c = \|e\times d\|\)

Let \(c=e\times d\). A small change in \(e\) gives
\[
\delta c = \delta e \times d.
\]
Then
\[
\delta r_c = \frac{c}{\|c\|}\cdot \delta c
= \hat c \cdot (\delta e\times d).
\]
Use the scalar triple product reordering:
\[
\hat c \cdot (\delta e\times d) = \delta e\cdot (d\times \hat c).
\]
So
\[
\nabla_e r_c = d\times \hat c = \frac{d\times c}{\|c\|}.
\]

Similarly, a change in \(d\) gives \(\delta c = e\times \delta d\), so
\[
\delta r_c = \hat c\cdot (e\times \delta d)=\delta d\cdot (\hat c\times e),
\]
hence
\[
\nabla_d r_c = \hat c\times e = \frac{c\times e}{\|c\|}.
\]

### 3.2 Now: derivative of \(r_c^\alpha\)
\[
\nabla_e r_c^\alpha = \alpha r_c^{\alpha-1}\nabla_e r_c
= \alpha r_c^{\alpha-2}(d\times c).
\]
\[
\nabla_d r_c^\alpha = \alpha r_c^{\alpha-2}(c\times e).
\]

### 3.3 Combine with the denominator \(r_d^{-\beta}\)
We also have
\[
\nabla_d r_d^{-\beta} = -\beta r_d^{-\beta-2} d.
\]

Finally:
\[
\boxed{
\nabla_e K(e,d)=\alpha \|e\times d\|^{\alpha-2}\,\|d\|^{-\beta}\,(d\times (e\times d))
}
\]
\[
\boxed{
\nabla_d K(e,d)=\alpha \|e\times d\|^{\alpha-2}\,\|d\|^{-\beta}\,((e\times d)\times e)
-\beta \|e\times d\|^\alpha\,\|d\|^{-\beta-2}\,d
}
\]

You can optionally expand the triple products:
- \(d\times(e\times d)=\|d\|^2 e-(d\cdot e)d\)
- \((e\times d)\times e=\|e\|^2 d-(d\cdot e)e\)

These expansions avoid explicitly computing cross-of-cross if you prefer.

---
""")

md(r"""
## 4. Assemble the gradient for one disjoint edge pair \((I,J)\)

Recall:
\[
E_{IJ}=\frac14\,\ell_J\,\ell_I^{1-\alpha}\sum_{i\in I}\sum_{j\in J}K(e_I,d_{ij}),
\qquad e_I=\gamma_{i_2}-\gamma_{i_1},\ d_{ij}=\gamma_i-\gamma_j.
\]
Let
\[
S_{IJ}:=\sum_{i\in I}\sum_{j\in J}K(e_I,d_{ij}).
\]
Then
\[
E_{IJ} = \frac14\,\ell_J\,\ell_I^{1-\alpha} S_{IJ}.
\]

### 4.1 Prefactor derivatives (lengths)
\[
\frac{\partial \ell_I}{\partial \gamma_{i_1}}=-T_I,\qquad
\frac{\partial \ell_I}{\partial \gamma_{i_2}}=+T_I.
\]
So
\[
\frac{\partial \ell_I^{1-\alpha}}{\partial \gamma_{i_1}}
=(1-\alpha)\ell_I^{-\alpha}\frac{\partial \ell_I}{\partial \gamma_{i_1}}
=(1-\alpha)\ell_I^{-\alpha}(-T_I)
=(1-\alpha)\ell_I^{-\alpha-1}(\gamma_{i_1}-\gamma_{i_2}).
\]
Similarly for \(i_2\) with opposite sign.

For edge \(J\),
\[
\frac{\partial \ell_J}{\partial \gamma_{j_1}}=-T_J,\qquad
\frac{\partial \ell_J}{\partial \gamma_{j_2}}=+T_J.
\]

### 4.2 Kernel derivatives distribute via \(e_I\) and \(d_{ij}\)

Important dependence signs:
- \(e_I=\gamma_{i_2}-\gamma_{i_1}\), so
  \(\partial e_I/\partial \gamma_{i_1}=-I\), \(\partial e_I/\partial \gamma_{i_2}=+I\).
- \(d_{ij}=\gamma_i-\gamma_j\), so
  \(\partial d_{ij}/\partial \gamma_i=+I\), \(\partial d_{ij}/\partial \gamma_j=-I\).

So for a single endpoint pair \((i,j)\) inside \((I,J)\),
\[
\frac{\partial K}{\partial \gamma_i} = \nabla_d K,
\qquad
\frac{\partial K}{\partial \gamma_j} = -\nabla_d K,
\]
and for the edge endpoints of \(I\),
\[
\frac{\partial K}{\partial \gamma_{i_1}} = -\nabla_e K \quad (\text{because } \partial e/\partial \gamma_{i_1}=-I),
\qquad
\frac{\partial K}{\partial \gamma_{i_2}} = +\nabla_e K.
\]
Crucially: **every** \((i,j)\) term contributes to both \(i_1\) and \(i_2\) via \(\nabla_e K\), because the same \(e_I\) is used for all four point samples.

### 4.3 Final assembly rule (algorithmic form)

For each disjoint edge pair \((I,J)\):
1. compute \(S_{IJ}=\sum K\),
2. add length-prefactor contributions to the four vertices,
3. for each of the 4 endpoint pairs \((i,j)\), add:
   - to vertex \(i\): \(+\nabla_d K\)
   - to vertex \(j\): \(-\nabla_d K\)
   - to \(i_1\): \(-\nabla_e K\)
   - to \(i_2\): \(+\nabla_e K\)
all multiplied by the common scalar prefactor \(\frac14\,\ell_J\,\ell_I^{1-\alpha}\).

That is exactly what you want to code.

---
""")

md(r"""
## 5. Concrete numbers (so you can “see” the quantities)

Let’s pick a simple configuration with two disjoint edges in 3D.

- Edge \(I=(i_1,i_2)\):
  \(\gamma_{i_1}=(0,0,0)\),
  \(\gamma_{i_2}=(1,0,0)\).
  So
  \(e=(1,0,0)\), \(\ell_I=1\), \(T_I=(1,0,0)\).

- Edge \(J=(j_1,j_2)\):
  \(\gamma_{j_1}=(0,1,1)\),
  \(\gamma_{j_2}=(1,1,1)\).
  So \(J\) is parallel to \(I\), shifted in \(y,z\).

Pick one endpoint pair: \(i=i_1\), \(j=j_1\).
Then
\[
d=\gamma_{i_1}-\gamma_{j_1}=(0,-1,-1),\quad \|d\|=\sqrt2.
\]
\[
c=e\times d=(1,0,0)\times(0,-1,-1)=(0,1,-1),\quad \|c\|=\sqrt2.
\]

If \(\alpha=3\), \(\beta=6\),
\[
K=\frac{\|c\|^3}{\|d\|^6}=\frac{(\sqrt2)^3}{(\sqrt2)^6}=2^{-3/2}\approx 0.353553.
\]

Now check the triple product pieces:
\[
d\times(e\times d)=\|d\|^2 e-(d\cdot e)d = 2(1,0,0)-0\cdot d=(2,0,0).
\]
\[
(e\times d)\times e=\|e\|^2 d-(d\cdot e)e = 1\cdot d - 0\cdot e = (0,-1,-1).
\]

So, up to scalar factors, \(\nabla_e K\) points along \(+x\) and \(\nabla_d K\) points along \(d\) (here).

Next we’ll compute this with code and visualize.

---
""")

code(r"""
import numpy as np
import matplotlib.pyplot as plt

def cross(a,b): return np.cross(a,b)
def dot(a,b): return float(np.dot(a,b))
def norm(a): return float(np.linalg.norm(a))

def kernel_and_grads(e, d, alpha=3.0, beta=6.0, eps=1e-10):
    # Matches the JS version: use eps by adding to norms, not inside squares
    rd = norm(d) + eps
    c = cross(e,d)
    rc = norm(c) + eps
    K = (rc**alpha) / (rd**beta)

    # Differentials of rc and rd (using rc without eps in denominators to avoid division by 0)
    rc_raw = norm(c)
    rd_raw = norm(d)

    if rc_raw < 1e-14:
        # fallback: zero out the rc-directional derivatives
        dc_dd = np.zeros(3)
        dc_de = np.zeros(3)
    else:
        # ∂||c||/∂d = (c×e)/||c|| ; ∂||c||/∂e = (d×c)/||c||
        dc_dd = cross(c,e) / rc_raw
        dc_de = cross(d,c) / rc_raw

    if rd_raw < 1e-14:
        dr_dd = np.zeros(3)
    else:
        dr_dd = d / rd_raw

    # dK/dd = alpha*rc^(a-1)/rd^b * d||c||/dd  - beta*rc^a/rd^(b+1) * d||d||/dd
    coeff_c = alpha * (rc**(alpha-1)) / (rd**beta)
    coeff_d = -beta * (rc**alpha) / (rd**(beta+1))

    dK_dd = coeff_c * dc_dd + coeff_d * dr_dd
    dK_de = coeff_c * dc_de  # only numerator depends on e

    return K, dK_de, dK_dd

# numeric example from Section 5
i1 = np.array([0.,0.,0.])
i2 = np.array([1.,0.,0.])
j1 = np.array([0.,1.,1.])
j2 = np.array([1.,1.,1.])
e = i2 - i1
d = i1 - j1

K, dK_de, dK_dd = kernel_and_grads(e,d,alpha=3,beta=6,eps=1e-10)
K, dK_de, dK_dd
""")

md(r"""
## 6. Visualization of geometry (edges, a displacement, and the cross vector)

We’ll draw:
- the two edges \(I\) and \(J\),
- the displacement \(d=\gamma_i-\gamma_j\) for one endpoint pair,
- the cross vector \(c=e\times d\) (perpendicular to the plane spanned by \(e\) and \(d\)).
""")

code(r"""
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_edges_and_vectors(i1,i2,j1,j2, pick_i=i1, pick_j=j1):
    e = i2 - i1
    d = pick_i - pick_j
    c = np.cross(e,d)

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')

    # edges
    ax.plot([i1[0],i2[0]],[i1[1],i2[1]],[i1[2],i2[2]], linewidth=4, label="Edge I")
    ax.plot([j1[0],j2[0]],[j1[1],j2[1]],[j1[2],j2[2]], linewidth=4, label="Edge J")

    # displacement d
    ax.quiver(pick_j[0], pick_j[1], pick_j[2], d[0], d[1], d[2], length=1.0, normalize=False, label="d (i-j)")

    # c = e x d anchored at pick_i
    ax.quiver(pick_i[0], pick_i[1], pick_i[2], c[0], c[1], c[2], length=1.0, normalize=False, label="c = e x d")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Two disjoint edges and one (d, c) pair")
    ax.legend(loc="upper left")
    plt.show()

plot_edges_and_vectors(i1,i2,j1,j2)
""")

md(r"""
## 7. From kernel gradients to vertex gradients (the “sign bookkeeping”)

This is the step that usually causes bugs.

### 7.1 One kernel term uses two “inputs”: \(e_I\) and \(d_{ij}\)
\[
K_{ij} = K(e_I, d_{ij}).
\]

So a vertex can influence \(K\) in two different ways:
- through the edge vector \(e_I=\gamma_{i_2}-\gamma_{i_1}\),
- through the point difference \(d_{ij}=\gamma_i-\gamma_j\).

### 7.2 The signs are determined by how \(e\) and \(d\) depend on vertices

- If you nudge \(\gamma_{i_1}\) in direction \(\delta\),
  then \(e_I\) changes by \(-\delta\).
  So
  \[
  \frac{\partial K}{\partial \gamma_{i_1}} \supset -\nabla_e K.
  \]

- If you nudge \(\gamma_{i_2}\) by \(\delta\),
  then \(e_I\) changes by \(+\delta\).
  So
  \[
  \frac{\partial K}{\partial \gamma_{i_2}} \supset +\nabla_e K.
  \]

- If you nudge \(\gamma_i\) by \(\delta\),
  then \(d_{ij}\) changes by \(+\delta\),
  so
  \[
  \frac{\partial K}{\partial \gamma_i} \supset +\nabla_d K.
  \]

- If you nudge \(\gamma_j\) by \(\delta\),
  then \(d_{ij}\) changes by \(-\delta\),
  so
  \[
  \frac{\partial K}{\partial \gamma_j} \supset -\nabla_d K.
  \]

**Important:** \(\nabla_e K\) contributes to both \(i_1\) and \(i_2\) **for every** endpoint sample \((i,j)\), because all samples share the same edge direction \(e_I\).

This is a common source of missing terms.

---
""")

md(r"""
## 8. Full energy + gradient for a tiny graph (and finite-difference verification)

We’ll implement:
- the discrete energy \(E\) for a set of vertices and edges,
- the analytical gradient,
- a finite-difference gradient check.

The check you should always do:
\[
\frac{\|g_{\text{analytic}} - g_{\text{FD}}\|}{\|g_{\text{FD}}\|} \ll 1.
\]

If it’s not tiny, it’s almost always:
- a sign mistake from cross-product ordering, or
- a missing “shared \(e_I\)” contribution, or
- mismatch in the energy’s exact discretization (extra factor of 2, etc.).
""")

code(r"""
def disjoint_pairs(edges):
    n = len(edges)
    out = [[] for _ in range(n)]
    for I,(a1,a2) in enumerate(edges):
        for J,(b1,b2) in enumerate(edges):
            if I==J: 
                continue
            if len({a1,a2,b1,b2})==4:
                out[I].append(J)
    return out

def energy(vertices, edges, dis, alpha=3.0, beta=6.0, eps=1e-10):
    V = np.asarray(vertices, dtype=float)
    total = 0.0
    for I,(i1,i2) in enumerate(edges):
        eI = V[i2]-V[i1]
        ellI = norm(eI)+eps
        for J in dis[I]:
            j1,j2 = edges[J]
            ellJ = norm(V[j2]-V[j1])+eps
            s = 0.0
            for i in (i1,i2):
                for j in (j1,j2):
                    d = V[i]-V[j]
                    rd = norm(d)+eps
                    rc = norm(np.cross(eI,d))+eps
                    s += (rc**alpha)/(rd**beta)
            total += 0.25 * (ellI**(1-alpha)) * ellJ * s
    return total/2.0  # matches your JS: divide by 2 because dis includes both directions

def grad_analytic(vertices, edges, dis, alpha=3.0, beta=6.0, eps=1e-10):
    V = np.asarray(vertices, dtype=float)
    g = np.zeros_like(V)

    def unit(v):
        r = np.linalg.norm(v)
        if r < 1e-14:
            return 0.0, np.zeros(3)
        return r, v/r

    for I,(i1,i2) in enumerate(edges):
        eI = V[i2]-V[i1]
        reI, eIhat = unit(eI)
        ellI = reI + eps
        ellI_pow = ellI**(1-alpha)

        # d(ellI^(1-a))/dv = (1-a)*ellI^(-a) * d(ellI)/dv
        dPow_i1 = (1-alpha)*(ellI**(-alpha))*(-eIhat)
        dPow_i2 = (1-alpha)*(ellI**(-alpha))*(+eIhat)

        for J in dis[I]:
            j1,j2 = edges[J]
            eJ = V[j2]-V[j1]
            reJ, eJhat = unit(eJ)
            ellJ = reJ + eps

            dEllJ_j1 = -eJhat
            dEllJ_j2 = +eJhat

            pairs = [(i1,j1),(i1,j2),(i2,j1),(i2,j2)]
            terms = []
            S = 0.0
            for i,j in pairs:
                d = V[i]-V[j]
                K, dK_de, dK_dd = kernel_and_grads(eI,d,alpha=alpha,beta=beta,eps=eps)
                S += K
                terms.append((i,j,dK_de,dK_dd))

            # prefactor paths
            g[i1] += 0.25*ellJ*S*dPow_i1
            g[i2] += 0.25*ellJ*S*dPow_i2

            g[j1] += 0.25*ellI_pow*S*dEllJ_j1
            g[j2] += 0.25*ellI_pow*S*dEllJ_j2

            base = 0.25*ellI_pow*ellJ
            for (i,j,dK_de,dK_dd) in terms:
                # d path
                g[i] += base*dK_dd
                g[j] -= base*dK_dd
                # e path (shared by all terms)
                g[i1] -= base*dK_de
                g[i2] += base*dK_de

    return g*0.5  # matches energy(...)/2

def grad_fd(vertices, edges, dis, alpha=3.0, beta=6.0, eps=1e-10, h=1e-6):
    V = np.asarray(vertices, dtype=float)
    g = np.zeros_like(V)
    E0 = energy(V, edges, dis, alpha=alpha, beta=beta, eps=eps)
    for vi in range(V.shape[0]):
        for k in range(3):
            Vp = V.copy()
            Vp[vi,k] += h
            E1 = energy(Vp, edges, dis, alpha=alpha, beta=beta, eps=eps)
            g[vi,k] = (E1 - E0)/h
    return g

# Test on the 2-edge example (4 vertices, 2 edges)
V = np.array([i1,i2,j1,j2])
edges = [(0,1),(2,3)]
dis = disjoint_pairs(edges)

Ea = energy(V, edges, dis)
ga = grad_analytic(V, edges, dis)
gf = grad_fd(V, edges, dis)

Ea, ga, gf, np.linalg.norm(ga-gf)/max(1e-12,np.linalg.norm(gf))
""")

md(r"""
If that final relative error is around ~1e-6 to 1e-8 (with forward differences), it’s good.
If you switch to *central* differences, it should often drop further (until floating-point noise dominates).

Let’s also sweep a parameter and watch energy/gradient behave smoothly.
""")

code(r"""
def sweep_distance(distances, alpha=3.0, beta=6.0):
    Es = []
    Gs = []
    for t in distances:
        V = np.array([
            [0.,0.,0.],
            [1.,0.,0.],
            [0.,t,1.],
            [1.,t,1.],
        ])
        edges=[(0,1),(2,3)]
        dis = disjoint_pairs(edges)
        Es.append(energy(V,edges,dis,alpha=alpha,beta=beta,eps=1e-10))
        Gs.append(np.linalg.norm(grad_analytic(V,edges,dis,alpha=alpha,beta=beta,eps=1e-10)))
    return np.array(Es), np.array(Gs)

ts = np.linspace(0.2, 4.0, 60)
Es, Gm = sweep_distance(ts)

plt.figure(figsize=(7,4))
plt.plot(ts, Es)
plt.xlabel("y-offset of edge J (distance grows)")
plt.ylabel("Energy")
plt.title("Energy decreases as edges separate")
plt.show()

plt.figure(figsize=(7,4))
plt.plot(ts, Gm)
plt.xlabel("y-offset of edge J")
plt.ylabel("||gradient||")
plt.title("Gradient magnitude vs separation")
plt.show()
""")

md(r"""
## 9. Implementation blueprint (what to code in C++/JS)

Here is the **minimal** mental model for a correct implementation:

### Inputs
- vertices \(V[i]\in\mathbb{R}^3\)
- edges \(I=(i_1,i_2)\)
- a list of disjoint edges for each \(I\): `disjointPairs[I]`

### For each disjoint edge pair \((I,J)\)
1. Compute \(e_I\), \(\ell_I\), \(\ell_I^{1-\alpha}\).
2. Compute \(e_J\), \(\ell_J\), \(T_J\).
3. For the 4 endpoint pairs \((i,j)\) with \(i\in\{i_1,i_2\}\), \(j\in\{j_1,j_2\}\):
   - compute \(d_{ij}\)
   - compute \(K_{ij}\)
   - compute \(\nabla_e K_{ij}\), \(\nabla_d K_{ij}\)
4. Compute \(S=\sum K_{ij}\).
5. Add length-prefactor gradient contributions:
   - to \(i_1,i_2\): derivative of \(\ell_I^{1-\alpha}\) times \(\ell_J S\)
   - to \(j_1,j_2\): derivative of \(\ell_J\) times \(\ell_I^{1-\alpha} S\)
6. Add kernel contributions for each term:
   - to vertex \(i\): \(+\nabla_d K\)
   - to vertex \(j\): \(-\nabla_d K\)
   - to vertex \(i_1\): \(-\nabla_e K\)
   - to vertex \(i_2\): \(+\nabla_e K\)
all multiplied by the common scalar \(0.25\,\ell_J\,\ell_I^{1-\alpha}\).

### Final factor-of-2 warning
If your energy loop includes both \((I,J)\) and \((J,I)\) (which is common when you precompute disjoint pairs naively), then you should divide energy by 2 **and** divide gradient by 2.  
If you sum only unordered pairs (e.g. \(J>I\)), you should not divide.

This must match between energy and gradient for finite-difference tests to pass.

---
""")

md(r"""
## 10. Common debugging checklist (fast)

If analytic vs finite-diff disagrees:

1. **Factor of 2 mismatch** (most common)
   - Are you counting both \((I,J)\) and \((J,I)\)?
   - Is your energy divided by 2 but your gradient not (or vice versa)?

2. **Missing shared-edge-vector term**
   - Every sample \(K_{ij}\) depends on the same \(e_I\).
   - Therefore, \(\nabla_e K_{ij}\) contributes to both \(i_1\) and \(i_2\) for all 4 samples.

3. **Cross product ordering sign**
   - \((a\times b) = -(b\times a)\).
   - A single swapped cross can flip an entire term’s sign.

4. **Degenerate geometry**
   - If \(\|e\times d\|\approx 0\), derivatives can get noisy.
   - Add a small \(\epsilon\) consistently, and test on non-degenerate random setups first.

5. **Mismatch in “epsilon convention”**
   - Adding \(\epsilon\) to the norm is not the same as adding \(\epsilon^2\) under the square root.
   - For gradient checks, match the exact convention used in energy.

---
""")

md(r"""
## 11. What you should be able to do now

- Start from \(K(e,d)=\|e\times d\|^\alpha/\|d\|^\beta\).
- Differentiate it by writing \(\delta c = \delta e\times d + e\times \delta d\).
- Use scalar triple product to isolate \(\delta e\) or \(\delta d\).
- Assemble vertex gradients by careful sign bookkeeping for \(e_I\) and \(d_{ij}\).
- Verify correctness with finite differences on random inputs.

If you can do those steps reliably, you can extend the method to:
- different sampling schemes,
- different kernels,
- surfaces (triangle–triangle interactions),
- acceleration structures (BVHs) for performance.

---
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
}


out_path = Path("./tangent_point_energy_gradient_tutorial.ipynb")
out_path.write_text(nbf.writes(nb), encoding="utf-8")
str(out_path)
