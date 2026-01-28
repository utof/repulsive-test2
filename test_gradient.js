/**
 * Test file to verify the analytical gradient implementation.
 * 
 * This compares the analytical gradient against finite differences to ensure correctness.
 * The test uses a simple curve configuration where we can easily verify the results.
 */

// Simple helper functions (duplicated here for standalone testing)
function cross3D(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
}

function cross2D(a, b) {
    return a[0] * b[1] - a[1] * b[0];
}

function dot(a, b) {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

function subtract(a, b) {
    return a.map((val, i) => val - b[i]);
}

function scale(s, v) {
    return v.map(x => s * x);
}

function norm(v) {
    return Math.sqrt(dot(v, v));
}

function add(a, b) {
    return a.map((val, i) => val + b[i]);
}

/**
 * Calculate disjoint edge pairs (edges that share no vertices)
 */
function calculateDisjointEdgePairs(edges) {
    const numEdges = edges.length;
    const disjointPairs = [];
    
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
function calculateEnergy(vertices, edges, alpha, beta, disjointPairs, dimension, epsilon) {
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
 * Calculate gradient using finite differences
 */
function calculateGradientFiniteDiff(vertices, edges, alpha, beta, disjointPairs, dimension, epsilon, h) {
    const gradient = vertices.map(v => new Array(dimension).fill(0));
    const E0 = calculateEnergy(vertices, edges, alpha, beta, disjointPairs, dimension, epsilon);
    
    for (let v = 0; v < vertices.length; v++) {
        for (let d = 0; d < dimension; d++) {
            // Perturb vertex
            const perturbedVertices = vertices.map(vtx => [...vtx]);
            perturbedVertices[v][d] += h;
            
            const E1 = calculateEnergy(perturbedVertices, edges, alpha, beta, disjointPairs, dimension, epsilon);
            gradient[v][d] = (E1 - E0) / h;
        }
    }
    
    return gradient;
}

/**
 * Analytical gradient (simplified implementation for testing)
 */
function calculateGradientAnalytical(vertices, edges, alpha, beta, disjointPairs, dimension, epsilon) {
    const gradient = vertices.map(() => new Array(dimension).fill(0));
    
    const addToGradient = (vertexIdx, contribution) => {
        for (let d = 0; d < dimension; d++) {
            gradient[vertexIdx][d] += contribution[d] || 0;
        }
    };
    
    for (let I = 0; I < edges.length; I++) {
        if (!disjointPairs[I]) continue;
        
        const [i1, i2] = edges[I];
        const e_I = subtract(vertices[i2], vertices[i1]);
        const ell_I = norm(e_I) + epsilon;
        const T_I = scale(1/ell_I, e_I);
        
        for (const J of disjointPairs[I]) {
            const [j1, j2] = edges[J];
            const e_J = subtract(vertices[j2], vertices[j1]);
            const ell_J = norm(e_J) + epsilon;
            const T_J = scale(1/ell_J, e_J);
            
            // Compute kernel values
            const pairs = [
                {i: i1, j: j1}, {i: i1, j: j2},
                {i: i2, j: j1}, {i: i2, j: j2}
            ];
            
            let sumK = 0;
            const kernelInfo = pairs.map(({i, j}) => {
                const d = subtract(vertices[i], vertices[j]);
                const d_norm = norm(d) + epsilon;
                
                let c_norm;
                if (dimension === 3) {
                    c_norm = norm(cross3D(e_I, d)) + epsilon;
                } else {
                    c_norm = Math.abs(cross2D(e_I, d)) + epsilon;
                }
                
                const K = Math.pow(c_norm, alpha) / Math.pow(d_norm, beta);
                sumK += K;
                
                return {i, j, d, d_norm, c_norm, K};
            });
            
            const ell_I_pow = Math.pow(ell_I, 1 - alpha);
            
            // ---- i1 contributions ----
            addToGradient(i1, scale((1-alpha) * Math.pow(ell_I, -alpha-1) * sumK * 0.25 * ell_J, 
                                    subtract(vertices[i1], vertices[i2])));
            
            for (const {i, j, d, d_norm, c_norm} of kernelInfo) {
                const d_cross_ed = subtract(scale(d_norm*d_norm, e_I), scale(dot(d, e_I), d));
                const factor = alpha * Math.pow(c_norm, alpha-2) / Math.pow(d_norm, beta);
                addToGradient(i1, scale(0.25 * ell_J * ell_I_pow * factor, d_cross_ed));
                
                if (i === i1) {
                    const e_cross_ed = subtract(scale(dot(e_I, d), e_I), scale(ell_I*ell_I, d));
                    addToGradient(i1, scale(0.25 * ell_J * ell_I_pow * factor, e_cross_ed));
                    
                    const d_factor = -beta * Math.pow(c_norm, alpha) / Math.pow(d_norm, beta+2);
                    addToGradient(i1, scale(0.25 * ell_J * ell_I_pow * d_factor, d));
                }
            }
            
            // ---- i2 contributions ----
            addToGradient(i2, scale((1-alpha) * Math.pow(ell_I, -alpha-1) * sumK * 0.25 * ell_J,
                                    subtract(vertices[i2], vertices[i1])));
            
            for (const {i, j, d, d_norm, c_norm} of kernelInfo) {
                const d_cross_ed = subtract(scale(d_norm*d_norm, e_I), scale(dot(d, e_I), d));
                const factor = alpha * Math.pow(c_norm, alpha-2) / Math.pow(d_norm, beta);
                addToGradient(i2, scale(-0.25 * ell_J * ell_I_pow * factor, d_cross_ed));
                
                if (i === i2) {
                    const e_cross_ed = subtract(scale(dot(e_I, d), e_I), scale(ell_I*ell_I, d));
                    addToGradient(i2, scale(0.25 * ell_J * ell_I_pow * factor, e_cross_ed));
                    
                    const d_factor = -beta * Math.pow(c_norm, alpha) / Math.pow(d_norm, beta+2);
                    addToGradient(i2, scale(0.25 * ell_J * ell_I_pow * d_factor, d));
                }
            }
            
            // ---- j1 contributions ----
            addToGradient(j1, scale(-0.25 * ell_I_pow * sumK, T_J));
            
            for (const {i, j, d, d_norm, c_norm} of kernelInfo) {
                if (j !== j1) continue;
                
                const e_cross_ed = subtract(scale(dot(e_I, d), e_I), scale(ell_I*ell_I, d));
                const factor = alpha * Math.pow(c_norm, alpha-2) / Math.pow(d_norm, beta);
                addToGradient(j1, scale(-0.25 * ell_J * ell_I_pow * factor, e_cross_ed));
                
                const d_factor = -beta * Math.pow(c_norm, alpha) / Math.pow(d_norm, beta+2);
                addToGradient(j1, scale(-0.25 * ell_J * ell_I_pow * d_factor, d));
            }
            
            // ---- j2 contributions ----
            addToGradient(j2, scale(0.25 * ell_I_pow * sumK, T_J));
            
            for (const {i, j, d, d_norm, c_norm} of kernelInfo) {
                if (j !== j2) continue;
                
                const e_cross_ed = subtract(scale(dot(e_I, d), e_I), scale(ell_I*ell_I, d));
                const factor = alpha * Math.pow(c_norm, alpha-2) / Math.pow(d_norm, beta);
                addToGradient(j2, scale(-0.25 * ell_J * ell_I_pow * factor, e_cross_ed));
                
                const d_factor = -beta * Math.pow(c_norm, alpha) / Math.pow(d_norm, beta+2);
                addToGradient(j2, scale(-0.25 * ell_J * ell_I_pow * d_factor, d));
            }
        }
    }
    
    return gradient;
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
        const gradAn = calculateGradientAnalytical(vertices2D, edges2D, alpha, beta, disjoint, 2, epsilon);
        
        console.log("Finite Diff gradient:");
        console.log(gradFD.map(g => g.map(x => x.toFixed(6))));
        
        console.log("\nAnalytical gradient:");
        console.log(gradAn.map(g => g.map(x => x.toFixed(6))));
        
        // Compute relative error
        let maxRelError = 0;
        for (let v = 0; v < vertices2D.length; v++) {
            for (let d = 0; d < 2; d++) {
                const fd = gradFD[v][d];
                const an = gradAn[v][d];
                const relErr = Math.abs(fd - an) / (Math.abs(fd) + 1e-10);
                maxRelError = Math.max(maxRelError, relErr);
            }
        }
        console.log(`\nMax relative error: ${maxRelError.toExponential(3)}`);
        console.log(maxRelError < 0.01 ? "✓ PASSED" : "✗ FAILED");
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
        const gradAn = calculateGradientAnalytical(vertices3D, edges3D, alpha, beta, disjoint, 3, epsilon);
        
        console.log("Finite Diff gradient:");
        gradFD.forEach((g, i) => console.log(`  v${i}: [${g.map(x => x.toFixed(6)).join(", ")}]`));
        
        console.log("\nAnalytical gradient:");
        gradAn.forEach((g, i) => console.log(`  v${i}: [${g.map(x => x.toFixed(6)).join(", ")}]`));
        
        let maxRelError = 0;
        for (let v = 0; v < vertices3D.length; v++) {
            for (let d = 0; d < 3; d++) {
                const fd = gradFD[v][d];
                const an = gradAn[v][d];
                const relErr = Math.abs(fd - an) / (Math.abs(fd) + 1e-10);
                maxRelError = Math.max(maxRelError, relErr);
            }
        }
        console.log(`\nMax relative error: ${maxRelError.toExponential(3)}`);
        console.log(maxRelError < 0.01 ? "✓ PASSED" : "✗ FAILED");
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
        const gradAn = calculateGradientAnalytical(vertices, edges, alpha, beta, disjoint, 3, epsilon);
        
        let maxRelError = 0;
        let maxAbsError = 0;
        for (let v = 0; v < vertices.length; v++) {
            for (let d = 0; d < 3; d++) {
                const fd = gradFD[v][d];
                const an = gradAn[v][d];
                const absErr = Math.abs(fd - an);
                const relErr = absErr / (Math.abs(fd) + 1e-10);
                maxRelError = Math.max(maxRelError, relErr);
                maxAbsError = Math.max(maxAbsError, absErr);
            }
        }
        
        console.log(`Max absolute error: ${maxAbsError.toExponential(3)}`);
        console.log(`Max relative error: ${maxRelError.toExponential(3)}`);
        console.log(maxRelError < 0.01 ? "✓ PASSED" : "✗ FAILED");
    }
    
    console.log("\n" + "=".repeat(50) + "\n");
    console.log("All tests completed!");
}

// Run the tests
runTests();