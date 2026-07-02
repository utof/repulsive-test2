/**
 * Test file to verify the analytical gradient implementation.
 *
 * This compares the analytical gradient against finite differences to ensure correctness.
 * The test uses a simple curve configuration where we can easily verify the results.
 */

import {
    calculateDisjointPairs,
    calculateEnergy,
    gradientAnalytical,
} from './src/core/tangentPointEnergy';
import type { Edge, Vec3 } from './src/core/testConfigs';

/**
 * Calculate gradient using CENTRAL finite differences, NOT forward.
 *
 * Central differences (E(+h) - E(-h)) / 2h are O(h^2) accurate, whereas a
 * forward difference (E(+h) - E(0)) / h is only O(h). The forward version
 * fabricates a spurious O(h) "gradient" wherever the true derivative is zero
 * (e.g. the out-of-plane z components of a planar configuration) and would
 * false-fail a correct analytical gradient — central differences cancel it.
 * @see docs/superpowers/specs/2026-07-01-tangent-point-hotpath-optimization-design.md — "Verification (the safety net)"
 */
function calculateGradientFiniteDiff(
    vertices: Vec3[],
    edges: Edge[],
    alpha: number,
    beta: number,
    disjointPairs: number[][],
    epsilon: number,
    h: number,
): Vec3[] {
    const gradient: Vec3[] = vertices.map(() => [0, 0, 0]);
    for (let v = 0; v < vertices.length; v++) {
        for (let d = 0; d < 3; d++) {
            const plus = vertices.map((vtx) => [...vtx] as Vec3);
            const minus = vertices.map((vtx) => [...vtx] as Vec3);
            plus[v][d] += h;
            minus[v][d] -= h;
            const Ep = calculateEnergy(plus, edges, disjointPairs, alpha, beta, epsilon);
            const Em = calculateEnergy(minus, edges, disjointPairs, alpha, beta, epsilon);
            gradient[v][d] = (Ep - Em) / (2 * h);
        }
    }
    return gradient;
}

/**
 * Compare two gradients with a combined absolute + relative tolerance:
 *   |fd - an| <= atol + rtol * max(|fd|, |an|)
 * NOT pure relative error.
 *
 * A pure relative-error test (absErr / |fd|) explodes when the true derivative
 * is ~0 — a tiny difference divided by a tiny magnitude false-fails. The mixed
 * tolerance is the standard robust gradient-check criterion. Intentional —
 * do not "simplify" to relative-only.
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
            if (absErr > maxAbs) {
                maxAbs = absErr;
                worst = { v, d, fd, an, absErr };
            }
            maxRel = Math.max(maxRel, relErr);
        }
    }

    return { ok, maxAbs, maxRel, worst };
}

// =============================================
// TEST CASES
// =============================================

function runTests() {
    console.log('=== Testing Analytical Gradient ===\n');

    // Test 1: Simple 2D square (4 vertices, 4 edges forming a square)
    console.log('Test 1: 2D Square');
    {
        const vertices2D: Vec3[] = [
            [0, 0, 0], // vertex 0
            [1, 0, 0], // vertex 1
            [1, 1, 0], // vertex 2
            [0, 1, 0], // vertex 3
        ];

        const edges2D: Edge[] = [
            [0, 1], // edge 0: bottom
            [1, 2], // edge 1: right
            [2, 3], // edge 2: top
            [3, 0], // edge 3: left
        ];

        const alpha = 3;
        const beta = 6;
        const epsilon = 1e-10;
        const h = 1e-6;

        const disjoint = calculateDisjointPairs(edges2D);

        const gradFD = calculateGradientFiniteDiff(
            vertices2D,
            edges2D,
            alpha,
            beta,
            disjoint,
            epsilon,
            h,
        );
        const gradAn = gradientAnalytical(vertices2D, edges2D, disjoint, alpha, beta, epsilon);

        console.log('Finite Diff gradient:');
        console.log(gradFD.map((g) => g.map((x) => x.toFixed(6))));

        console.log('\nAnalytical gradient:');
        console.log(gradAn.map((g) => g.map((x) => x.toFixed(6))));

        const check = checkGradients(gradFD, gradAn);
        console.log(
            `\nMax abs error: ${check.maxAbs.toExponential(3)}, max rel error: ${check.maxRel.toExponential(3)}`,
        );
        console.log(check.ok ? '✓ PASSED' : '✗ FAILED');
    }

    console.log('\n' + '='.repeat(50) + '\n');

    // Test 2: 3D curve with 5 vertices
    console.log('Test 2: 3D Zigzag curve');
    {
        const vertices3D: Vec3[] = [
            [0, 0, 0],
            [1, 0, 0],
            [1.5, 1, 0.5],
            [0.5, 1.5, 0],
            [0, 1, -0.5],
        ];

        const edges3D: Edge[] = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
        ];

        const alpha = 3;
        const beta = 6;
        const epsilon = 1e-10;
        const h = 1e-6;

        const disjoint = calculateDisjointPairs(edges3D);

        const gradFD = calculateGradientFiniteDiff(
            vertices3D,
            edges3D,
            alpha,
            beta,
            disjoint,
            epsilon,
            h,
        );
        const gradAn = gradientAnalytical(vertices3D, edges3D, disjoint, alpha, beta, epsilon);

        console.log('Finite Diff gradient:');
        gradFD.forEach((g, i) =>
            console.log(`  v${i}: [${g.map((x) => x.toFixed(6)).join(', ')}]`),
        );

        console.log('\nAnalytical gradient:');
        gradAn.forEach((g, i) =>
            console.log(`  v${i}: [${g.map((x) => x.toFixed(6)).join(', ')}]`),
        );

        const check = checkGradients(gradFD, gradAn);
        console.log(
            `\nMax abs error: ${check.maxAbs.toExponential(3)}, max rel error: ${check.maxRel.toExponential(3)}`,
        );
        console.log(check.ok ? '✓ PASSED' : '✗ FAILED');
    }

    console.log('\n' + '='.repeat(50) + '\n');

    // Test 3: Different alpha/beta values
    console.log('Test 3: Testing with α=2, β=4.5');
    {
        const vertices: Vec3[] = [
            [0, 0, 0],
            [1, 0.1, 0],
            [2, 0, 0],
            [0, 2, 0],
            [1, 2.1, 0],
            [2, 2, 0],
        ];

        const edges: Edge[] = [
            [0, 1],
            [1, 2], // bottom row
            [3, 4],
            [4, 5], // top row
        ];

        const alpha = 2;
        const beta = 4.5;
        const epsilon = 1e-10;
        const h = 1e-6;

        const disjoint = calculateDisjointPairs(edges);

        const gradFD = calculateGradientFiniteDiff(
            vertices,
            edges,
            alpha,
            beta,
            disjoint,
            epsilon,
            h,
        );
        const gradAn = gradientAnalytical(vertices, edges, disjoint, alpha, beta, epsilon);

        const check = checkGradients(gradFD, gradAn);

        console.log('Finite Diff gradient:');
        gradFD.forEach((g, i) =>
            console.log(`  v${i}: [${g.map((x) => x.toFixed(6)).join(', ')}]`),
        );

        console.log('\nAnalytical gradient:');
        gradAn.forEach((g, i) =>
            console.log(`  v${i}: [${g.map((x) => x.toFixed(6)).join(', ')}]`),
        );

        const { v, d, fd, an, absErr } = check.worst;
        console.log(
            `\nWorst component: v${v} dim ${d} (FD=${fd.toExponential(3)}, An=${an.toExponential(3)}, AbsErr=${absErr.toExponential(3)})`,
        );
        console.log(`Max absolute error: ${check.maxAbs.toExponential(3)}`);
        console.log(`Max relative error: ${check.maxRel.toExponential(3)}`);
        console.log(check.ok ? '✓ PASSED' : '✗ FAILED');
    }

    console.log('\n' + '='.repeat(50) + '\n');
    console.log('All tests completed!');
}

// Run the tests
runTests();
