/**
 * Test file to verify the analytical gradient implementation.
 * 
 * This compares the analytical gradient against finite differences to ensure correctness.
 * The test uses a simple curve configuration where we can easily verify the results.
 */
import { testConfigs, type GraphState, type Vec3, type Edge } from './src/testConfigs';

// Simple helper functions (duplicated here for standalone testing)
function cross3D(a: number[], b: number[]): number[] {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
}

function cross2D(a: number[], b: number[]): number {
    return a[0] * b[1] - a[1] * b[0];
}

function dot(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

function subtract(a: number[], b: number[]): number[] {
    return a.map((val, i) => val - b[i]);
}

function scale(s: number, v: number[]): number[] {
    return v.map(x => s * x);
}

function norm(v: number[]): number {
    return Math.sqrt(dot(v, v));
}

function add(a: number[], b: number[]): number[] {
    return a.map((val, i) => val + b[i]);
}

/**
 * Calculate disjoint edge pairs (edges that share no vertices)
 */
function calculateDisjointEdgePairs(edges: number[][]): number[][] {
    const numEdges = edges.length;
    const disjointPairs: number[][] = [];
    
    for (let i = 0; i < numEdges; i++) {
        disjointPairs[i] = [];
        for (let j = 0; j < numEdges; j++) {
            if (i === j) continue;
            
            const edge1 = edges[i];
            const edge2 = edges[j];
            
            // Check if edges share any vertex
            if (edge1[0] !== edge2[0] && edge1[0] !== edge2[1] &&
                edge1[1] !== edge2[0] && edge1[1] !== edge2[1]) {
                disjointPairs[i].push(j);
            }
        }
    }
    return disjointPairs;
}

/**
 * Calculate the discrete energy (for verification)
 */
function calculateEnergy(vertices: number[][], edges: number[][], alpha: number, beta: number, disjointPairs: number[][], dimension: number, epsilon: number): number {
    let totalEnergy = 0;
    
    for (let I = 0; I < edges.length; I++) {
        if (!disjointPairs[I]) continue;
        
        const [i1, i2] = edges[I];
        const e_I = subtract(vertices[i2], vertices[i1]);
        const ell_I = norm(e_I) + epsilon;
        
        for (const J of disjointPairs[I]) {
            const [j1, j2] = edges[J];
            const ell_J = norm(subtract(vertices[j2], vertices[j1])) + epsilon;
            
            let sumK = 0;
            for (const i of [i1, i2]) {
                for (const j of [j1, j2]) {
                    const d = subtract(vertices[i], vertices[j]);
                    const d_norm = norm(d) + epsilon;
                    
                    let c_norm;
                    if (dimension === 3) {
                        c_norm = norm(cross3D(e_I, d)) + epsilon;
                    } else {
                        c_norm = Math.abs(cross2D(e_I, d)) + epsilon;
                    }
                    
                    sumK += Math.pow(c_norm, alpha) / Math.pow(d_norm, beta);
                }
            }
            
            totalEnergy += 0.25 * Math.pow(ell_I, 1 - alpha) * ell_J * sumK;
        }
    }
    
    return totalEnergy / 2; // Divide by 2 for symmetry
}

/**
 * Calculate gradient using central finite differences.
 *
 * Central differences (E(+h) - E(-h)) / 2h are O(h^2) accurate, whereas a
 * forward difference (E(+h) - E(0)) / h is only O(h). The forward version
 * produces a spurious O(h) "gradient" wherever the true derivative is zero
 * (e.g. the out-of-plane z components of a planar configuration) — a
 * finite-difference artifact, not a real gradient. Central differences cancel it.
 */
function calculateGradientFiniteDiff(vertices: number[][], edges: number[][], alpha: number, beta: number, disjointPairs: number[][], dimension: number, epsilon: number, h: number): number[][] {
    const gradient = vertices.map(v => new Array(dimension).fill(0));

    for (let v = 0; v < vertices.length; v++) {
        for (let d = 0; d < dimension; d++) {
            const plus = vertices.map(vtx => [...vtx]);
            const minus = vertices.map(vtx => [...vtx]);
            plus[v][d] += h;
            minus[v][d] -= h;

            const Eplus = calculateEnergy(plus, edges, alpha, beta, disjointPairs, dimension, epsilon);
            const Eminus = calculateEnergy(minus, edges, alpha, beta, disjointPairs, dimension, epsilon);
            gradient[v][d] = (Eplus - Eminus) / (2 * h);
        }
    }

    return gradient;
}

/**
 * Analytical gradient for the same energy as calculateEnergy():
 *  E = (1/2) * Σ_I Σ_{J in disjoint(I)} 0.25 * (||e_I||+eps)^(1-a) * (||e_J||+eps) * Σ_{i∈I} Σ_{j∈J} ( (||e_I × d||+eps)^a / (||d||+eps)^b )
 *
 * Supports dimension = 2 or 3.
 * - In 2D it matches calculateEnergy's use of abs(cross2D(e, d)).
 * - Returns gradients with the same dimension as vertices: [x,y] or [x,y,z].
 */
function gradientAnalytical(
  vertices: number[][],
  edges: number[][],
  disjointPairs: number[][],
  alpha: number,
  beta: number,
  epsilon: number,
  dimension: 2 | 3 = 3
): number[][] {
  const gradient: number[][] = vertices.map(() => new Array(dimension).fill(0));

  // --- small vector helpers that respect `dimension`
  const dotN = (a: number[], b: number[]) => {
    let s = 0;
    for (let k = 0; k < dimension; k++) s += (a[k] || 0) * (b[k] || 0);
    return s;
  };

  const addN = (a: number[], b: number[]) => a.map((v, i) => v + (b[i] || 0));
  const subN = (a: number[], b: number[]) => a.map((v, i) => v - (b[i] || 0));
  const scaleN = (s: number, v: number[]) => v.map((x) => s * x);

  const normN = (v: number[]) => Math.sqrt(dotN(v, v));

  const addToGrad = (idx: number, v: number[]) => {
    for (let k = 0; k < dimension; k++) gradient[idx][k] += v[k] || 0;
  };

  const safeUnit = (v: number[]) => {
    const r = normN(v);
    if (r < 1e-14) return { len: r, unit: new Array(dimension).fill(0) };
    return { len: r, unit: scaleN(1 / r, v) };
  };

  // 3D cross (returns Vec3)
  const cross3 = (a: number[], b: number[]) => ([
    (a[1] || 0) * (b[2] || 0) - (a[2] || 0) * (b[1] || 0),
    (a[2] || 0) * (b[0] || 0) - (a[0] || 0) * (b[2] || 0),
    (a[0] || 0) * (b[1] || 0) - (a[1] || 0) * (b[0] || 0),
  ]);

  // 2D "cross" scalar (z-component of 3D cross when z=0)
  const cross2 = (a: number[], b: number[]) => (a[0] || 0) * (b[1] || 0) - (a[1] || 0) * (b[0] || 0);

  /**
   * Kernel:
   *   f(d,e) = (C + eps)^alpha / (D + eps)^beta
   * where:
   *   D = ||d||
   *   C = ||e × d|| in 3D
   *   C = |cross2D(e,d)| in 2D  (matches your calculateEnergy)
   *
   * Returns:
   *  f
   *  df_dd : gradient w.r.t. d-vector
   *  df_de : gradient w.r.t. e-vector
   */
  const kernelDerivs = (e: number[], dvec: number[]) => {
    const { len: rd, unit: dHat } = safeUnit(dvec);
    const d_eps = rd + epsilon;

    let C: number;            // "cross magnitude" used in energy (before +eps)
    let dC_dd: number[];      // ∂C/∂dvec
    let dC_de: number[];      // ∂C/∂evec

    if (dimension === 3) {
      const cvec = cross3(e, dvec);
      const rc = Math.sqrt((cvec[0] ** 2) + (cvec[1] ** 2) + (cvec[2] ** 2));
      C = rc;

      // If rc == 0, the direction is undefined; we zero it (consistent with your safe handling).
      if (rc < 1e-14) {
        dC_dd = [0, 0, 0];
        dC_de = [0, 0, 0];
      } else {
        // d||e×d||/dd = ((e×d)×e)/||e×d||
        dC_dd = scaleN(1 / rc, cross3(cvec, e));
        // d||e×d||/de = (d×(e×d))/||e×d||
        dC_de = scaleN(1 / rc, cross3(dvec, cvec));
      }
    } else {
      // 2D case: C = |cross2D(e,d)|
      const c = cross2(e, dvec);
      const s = c > 0 ? 1 : c < 0 ? -1 : 0; // derivative of abs at 0 -> 0 (matches "safe" behavior)
      C = Math.abs(c);

      // ∂(cross2D(e,d))/∂d = [-e_y, e_x]
      const dc_dd = [-(e[1] || 0), (e[0] || 0)];
      // ∂(cross2D(e,d))/∂e = [d_y, -d_x]
      const dc_de = [(dvec[1] || 0), -(dvec[0] || 0)];

      dC_dd = scaleN(s, dc_dd);
      dC_de = scaleN(s, dc_de);
    }

    const c_eps = C + epsilon;

    const cPowA = Math.pow(c_eps, alpha);
    const dPowB = Math.pow(d_eps, beta);
    const f = cPowA / dPowB;

    // df/dd = alpha*(c_eps)^(a-1)/d_eps^b * dC/dd  - beta*(c_eps)^a/d_eps^(b+1) * dHat
    const coeff_c = alpha * Math.pow(c_eps, alpha - 1) / dPowB;
    const coeff_d = -beta * cPowA / Math.pow(d_eps, beta + 1);

    const df_dd = addN(scaleN(coeff_c, dC_dd), scaleN(coeff_d, dHat));
    const df_de = scaleN(coeff_c, dC_de);

    return { f, df_dd, df_de };
  };

  for (let I = 0; I < edges.length; I++) {
    const dis = disjointPairs[I];
    if (!dis || dis.length === 0) continue;

    const [i1, i2] = edges[I];

    const e_I = subN(vertices[i2], vertices[i1]);
    const { len: reI, unit: eI_hat } = safeUnit(e_I);
    const ell_I = reI + epsilon;
    const ell_I_pow = Math.pow(ell_I, 1 - alpha);

    // d(ell_I^(1-a))/dv = (1-a)*ell_I^(-a) * d(ell_I)/dv
    const commonPow = (1 - alpha) * Math.pow(ell_I, -alpha);
    const dPow_dv_i1 = scaleN(commonPow, scaleN(-1, eI_hat));
    const dPow_dv_i2 = scaleN(commonPow, eI_hat);

    for (const J of dis) {
      const [j1, j2] = edges[J];

      const e_J = subN(vertices[j2], vertices[j1]);
      const { unit: eJ_hat } = safeUnit(e_J);
      const ell_J = normN(e_J) + epsilon;

      const dEllJ_dv_j1 = scaleN(-1, eJ_hat);
      const dEllJ_dv_j2 = eJ_hat;

      const pairs = [
        { i: i1, j: j1 },
        { i: i1, j: j2 },
        { i: i2, j: j1 },
        { i: i2, j: j2 },
      ];

      let sumF = 0;
      const terms: { i: number; j: number; df_dd: number[]; df_de: number[] }[] = [];

      for (const { i, j } of pairs) {
        const dvec = subN(vertices[i], vertices[j]);
        const { f, df_dd, df_de } = kernelDerivs(e_I, dvec);
        sumF += f;
        terms.push({ i, j, df_dd, df_de });
      }

      // E_IJ = 0.25 * ell_I^(1-a) * ell_J * sumF

      // via ell_I^(1-a)
      addToGrad(i1, scaleN(0.25 * ell_J * sumF, dPow_dv_i1));
      addToGrad(i2, scaleN(0.25 * ell_J * sumF, dPow_dv_i2));

      // via ell_J
      addToGrad(j1, scaleN(0.25 * ell_I_pow * sumF, dEllJ_dv_j1));
      addToGrad(j2, scaleN(0.25 * ell_I_pow * sumF, dEllJ_dv_j2));

      const base = 0.25 * ell_I_pow * ell_J;

      for (const t of terms) {
        // dvec = v_i - v_j
        addToGrad(t.i, scaleN(base, t.df_dd));
        addToGrad(t.j, scaleN(-base, t.df_dd));

        // e_I = v_i2 - v_i1
        addToGrad(i1, scaleN(-base, t.df_de));
        addToGrad(i2, scaleN(base, t.df_de));
      }
    }
  }

  // calculateEnergy divides by 2 for symmetry, so gradient must match
  for (let v = 0; v < gradient.length; v++) {
    for (let k = 0; k < dimension; k++) gradient[v][k] *= 0.5;
  }

  return gradient;
}


/**
 * Compare two gradients with a combined absolute + relative tolerance:
 *   |fd - an| <= atol + rtol * max(|fd|, |an|)
 *
 * A pure relative-error test (absErr / |fd|) explodes when the true derivative
 * is ~0 — a tiny difference divided by a tiny magnitude. The mixed tolerance
 * is the standard robust gradient-check criterion.
 */
function checkGradients(gradFD: number[][], gradAn: number[][], atol = 1e-6, rtol = 1e-5) {
    let ok = true;
    let maxAbs = 0;
    let maxRel = 0;
    let worst = { v: -1, d: -1, fd: 0, an: 0, absErr: 0 };

    for (let v = 0; v < gradFD.length; v++) {
        for (let d = 0; d < gradFD[v].length; d++) {
            const fd = gradFD[v][d];
            const an = gradAn[v][d];
            const absErr = Math.abs(fd - an);
            const relErr = absErr / Math.max(Math.abs(fd), Math.abs(an), 1e-12);

            if (absErr > atol + rtol * Math.max(Math.abs(fd), Math.abs(an))) ok = false;
            if (absErr > maxAbs) { maxAbs = absErr; worst = { v, d, fd, an, absErr }; }
            maxRel = Math.max(maxRel, relErr);
        }
    }

    return { ok, maxAbs, maxRel, worst };
}

// =============================================
// TEST CASES
// =============================================

function runTests() {
    console.log("=== Testing Analytical Gradient ===\n");
    
    // Test 1: Simple 2D square (4 vertices, 4 edges forming a square)
    console.log("Test 1: 2D Square");
    {
        const vertices2D = [
            [0, 0],    // vertex 0
            [1, 0],    // vertex 1
            [1, 1],    // vertex 2
            [0, 1]     // vertex 3
        ];
        
        const edges2D = [
            [0, 1],  // edge 0: bottom
            [1, 2],  // edge 1: right
            [2, 3],  // edge 2: top
            [3, 0]   // edge 3: left
        ];
        
        const alpha = 3;
        const beta = 6;
        const epsilon = 1e-10;
        const h = 1e-6;
        
        const disjoint = calculateDisjointEdgePairs(edges2D);
        
        const gradFD = calculateGradientFiniteDiff(vertices2D, edges2D, alpha, beta, disjoint, 2, epsilon, h);
        const gradAn = gradientAnalytical(vertices2D, edges2D, disjoint, alpha, beta, epsilon, 2);
        
        console.log("Finite Diff gradient:");
        console.log(gradFD.map(g => g.map(x => x.toFixed(6))));
        
        console.log("\nAnalytical gradient:");
        console.log(gradAn.map(g => g.map(x => x.toFixed(6))));
        
        const check = checkGradients(gradFD, gradAn);
        console.log(`\nMax abs error: ${check.maxAbs.toExponential(3)}, max rel error: ${check.maxRel.toExponential(3)}`);
        console.log(check.ok ? "✓ PASSED" : "✗ FAILED");
    }
    
    console.log("\n" + "=".repeat(50) + "\n");
    
    // Test 2: 3D curve with 5 vertices
    console.log("Test 2: 3D Zigzag curve");
    {
        const vertices3D = [
            [0, 0, 0],
            [1, 0, 0],
            [1.5, 1, 0.5],
            [0.5, 1.5, 0],
            [0, 1, -0.5]
        ];
        
        const edges3D = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4]
        ];
        
        const alpha = 3;
        const beta = 6;
        const epsilon = 1e-10;
        const h = 1e-6;
        
        const disjoint = calculateDisjointEdgePairs(edges3D);
        
        const gradFD = calculateGradientFiniteDiff(vertices3D, edges3D, alpha, beta, disjoint, 3, epsilon, h);
        const gradAn = gradientAnalytical(vertices3D, edges3D, disjoint, alpha, beta, epsilon, 3);
        
        console.log("Finite Diff gradient:");
        gradFD.forEach((g, i) => console.log(`  v${i}: [${g.map(x => x.toFixed(6)).join(", ")}]`));
        
        console.log("\nAnalytical gradient:");
        gradAn.forEach((g, i) => console.log(`  v${i}: [${g.map(x => x.toFixed(6)).join(", ")}]`));
        
        const check = checkGradients(gradFD, gradAn);
        console.log(`\nMax abs error: ${check.maxAbs.toExponential(3)}, max rel error: ${check.maxRel.toExponential(3)}`);
        console.log(check.ok ? "✓ PASSED" : "✗ FAILED");
    }
    
    console.log("\n" + "=".repeat(50) + "\n");
    
    // Test 3: Different alpha/beta values
    console.log("Test 3: Testing with α=2, β=4.5");
    {
        const vertices = [
            [0, 0, 0],
            [1, 0.1, 0],
            [2, 0, 0],
            [0, 2, 0],
            [1, 2.1, 0],
            [2, 2, 0]
        ];
        
        const edges = [
            [0, 1], [1, 2],  // bottom row
            [3, 4], [4, 5]   // top row
        ];
        
        const alpha = 2;
        const beta = 4.5;
        const epsilon = 1e-10;
        const h = 1e-6;
        
        const disjoint = calculateDisjointEdgePairs(edges);
        
        const gradFD = calculateGradientFiniteDiff(vertices, edges, alpha, beta, disjoint, 3, epsilon, h);
        const gradAn = gradientAnalytical(vertices, edges, disjoint, alpha, beta, epsilon, 3);
        
        const check = checkGradients(gradFD, gradAn);

        console.log("Finite Diff gradient:");
        gradFD.forEach((g, i) => console.log(`  v${i}: [${g.map(x => x.toFixed(6)).join(", ")}]`));

        console.log("\nAnalytical gradient:");
        gradAn.forEach((g, i) => console.log(`  v${i}: [${g.map(x => x.toFixed(6)).join(", ")}]`));

        const { v, d, fd, an, absErr } = check.worst;
        console.log(`\nWorst component: v${v} dim ${d} (FD=${fd.toExponential(3)}, An=${an.toExponential(3)}, AbsErr=${absErr.toExponential(3)})`);
        console.log(`Max absolute error: ${check.maxAbs.toExponential(3)}`);
        console.log(`Max relative error: ${check.maxRel.toExponential(3)}`);
        console.log(check.ok ? "✓ PASSED" : "✗ FAILED");
    }
    
    console.log("\n" + "=".repeat(50) + "\n");
    console.log("All tests completed!");
}

// Run the tests
runTests();