// analyticalDifferential.js
// 
// Analytical gradient computation for the tangent-point energy.
// This replaces the finite-difference approximation with exact derivatives.
//
// Reference: "Repulsive Curves" by Yu, Schumacher, and Crane (2020)
// See tangent_point_gradient_derivation.md for the mathematical derivation.

import * as math from 'mathjs';

/**
 * Compute the skew-symmetric matrix [v]_× such that [v]_× * w = v × w
 * 
 * For v = [v1, v2, v3], the matrix is:
 *     [  0  -v3   v2 ]
 *     [ v3   0  -v1 ]
 *     [-v2  v1   0  ]
 * 
 * @param {Array} v - 3D vector
 * @returns {Array} - 3x3 skew-symmetric matrix as 2D array
 */
function skewSymmetric(v) {
    return [
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ];
}

/**
 * Cross product of two 3D vectors
 * @param {Array} a - First vector
 * @param {Array} b - Second vector
 * @returns {Array} - a × b
 */
function cross3D(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
}

/**
 * Dot product of two vectors
 * @param {Array} a - First vector
 * @param {Array} b - Second vector
 * @returns {number} - a · b
 */
function dot(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 * Vector subtraction
 * @param {Array} a - First vector
 * @param {Array} b - Second vector
 * @returns {Array} - a - b
 */
function subtract(a, b) {
    return a.map((val, i) => val - b[i]);
}

/**
 * Vector addition
 * @param {Array} a - First vector
 * @param {Array} b - Second vector
 * @returns {Array} - a + b
 */
function add(a, b) {
    return a.map((val, i) => val + b[i]);
}

/**
 * Scalar multiplication
 * @param {number} s - Scalar
 * @param {Array} v - Vector
 * @returns {Array} - s * v
 */
function scale(s, v) {
    return v.map(x => s * x);
}

/**
 * Vector norm (Euclidean length)
 * @param {Array} v - Vector
 * @returns {number} - |v|
 */
function norm(v) {
    return Math.sqrt(dot(v, v));
}

/**
 * Cross product for 2D (returns scalar: the z-component of 3D cross product)
 * @param {Array} a - 2D vector
 * @param {Array} b - 2D vector
 * @returns {number} - a₁b₂ - a₂b₁
 */
function cross2D(a, b) {
    return a[0] * b[1] - a[1] * b[0];
}

/**
 * Compute the analytical differential of the tangent-point energy
 * 
 * Mathematical Background:
 * The energy for an edge pair (I, J) is:
 *   E_IJ = (1/4) * ℓ_J * ℓ_I^(1-α) * Σ_{i∈I, j∈J} |ẽ_IJ|^α / |d_ij|^β
 * 
 * where:
 *   - ẽ_ij = e_I × d_ij (unnormalized cross product)
 *   - e_I = γ_{i₂} - γ_{i₁} (edge vector)
 *   - d_ij = γ_i - γ_j (difference vector)
 *   - ℓ_I = |e_I| (edge length)
 * 
 * @param {Array} vertices - Array of vertex positions, each is [x, y] or [x, y, z]
 * @param {Array} edges - Array of edges, each is [vertexIndex1, vertexIndex2]
 * @param {number} alpha - Energy parameter α (typically 2-3)
 * @param {number} beta - Energy parameter β (typically 4-6)
 * @param {Array} disjointPairs - Pre-computed array where disjointPairs[i] lists 
 *                                 indices of edges disjoint from edge i
 * @param {Object} config - Configuration object with:
 *                          - dimension: 2 or 3
 *                          - epsilonKernel: small value to prevent division by zero
 * @returns {Array} - Gradient array, same shape as vertices
 */
export function calculateAnalyticalDifferential(vertices, edges, alpha, beta, disjointPairs, config) {
    const numVertices = vertices.length;
    const numEdges = edges.length;
    const dimension = config?.dimension || 3;
    const epsilon = config?.epsilonKernel || 1e-10;
    
    // Initialize gradient to zero for all vertices
    const gradient = [];
    for (let i = 0; i < numVertices; i++) {
        gradient[i] = new Array(dimension).fill(0);
    }
    
    // Build adjacency: for each vertex, which edges contain it?
    const vertexToEdges = [];
    for (let v = 0; v < numVertices; v++) {
        vertexToEdges[v] = [];
    }
    for (let e = 0; e < numEdges; e++) {
        vertexToEdges[edges[e][0]].push({ edgeIndex: e, position: 0 }); // position 0 = first vertex
        vertexToEdges[edges[e][1]].push({ edgeIndex: e, position: 1 }); // position 1 = second vertex
    }
    
    // Pre-compute edge properties
    const edgeVectors = [];  // e_I = γ_{i₂} - γ_{i₁}
    const edgeLengths = [];  // ℓ_I = |e_I|
    const unitTangents = []; // T_I = e_I / ℓ_I
    
    for (let e = 0; e < numEdges; e++) {
        const [i1, i2] = edges[e];
        const v1 = vertices[i1];
        const v2 = vertices[i2];
        
        const eVec = subtract(v2, v1);
        const len = norm(eVec) + epsilon;
        const tangent = scale(1/len, eVec);
        
        edgeVectors[e] = eVec;
        edgeLengths[e] = len;
        unitTangents[e] = tangent;
    }
    
    // Process each pair of disjoint edges
    for (let I = 0; I < numEdges; I++) {
        if (!disjointPairs[I]) continue;
        
        const [i1, i2] = edges[I];
        const e_I = edgeVectors[I];
        const ell_I = edgeLengths[I];
        const T_I = unitTangents[I];
        
        for (const J of disjointPairs[I]) {
            if (J <= I) continue;  // Process each pair only once
            
            const [j1, j2] = edges[J];
            const e_J = edgeVectors[J];
            const ell_J = edgeLengths[J];
            const T_J = unitTangents[J];
            
            // Compute kernel values for all 4 vertex combinations
            // K_ij = |ẽ_ij|^α / |d_ij|^β  where ẽ_ij = e_I × d_ij
            const kernelData = [];
            
            for (const i of [i1, i2]) {
                for (const j of [j1, j2]) {
                    const d_ij = subtract(vertices[i], vertices[j]);
                    const d_norm = norm(d_ij) + epsilon;
                    
                    // Compute cross product (handles 2D and 3D)
                    let c_tilde;  // ẽ_ij = e_I × d_ij
                    let c_norm;
                    
                    if (dimension === 3) {
                        c_tilde = cross3D(e_I, d_ij);
                        c_norm = norm(c_tilde) + epsilon;
                    } else {
                        // For 2D, cross product gives a scalar (z-component)
                        const crossScalar = cross2D(e_I, d_ij);
                        c_norm = Math.abs(crossScalar) + epsilon;
                        c_tilde = [0, 0, crossScalar]; // Embed in 3D for consistent math
                    }
                    
                    const K = Math.pow(c_norm, alpha) / Math.pow(d_norm, beta);
                    
                    kernelData.push({
                        i, j,
                        d_ij, d_norm,
                        c_tilde, c_norm,
                        K
                    });
                }
            }
            
            // Sum of all kernel values for this edge pair
            const sumK = kernelData.reduce((sum, k) => sum + k.K, 0);
            
            // Coefficient from the outer factor: (1/4) * ℓ_J * ℓ_I^(1-α)
            const ell_I_pow = Math.pow(ell_I, 1 - alpha);
            const coeff = 0.25 * ell_J * ell_I_pow;
            
            // ============================================================
            // CASE 1: Derivatives with respect to vertices of edge I
            // ============================================================
            
            // ----- Vertex i1 (first endpoint of I) -----
            {
                // Term 1: Derivative of ℓ_I^(1-α)
                // d(ℓ_I^(1-α))/d(γ_{i₁}) = (1-α) * ℓ_I^(-α-1) * (γ_{i₁} - γ_{i₂})
                const term1_coeff = (1 - alpha) * Math.pow(ell_I, -alpha - 1) * sumK * 0.25 * ell_J;
                const term1 = scale(term1_coeff, subtract(vertices[i1], vertices[i2]));
                
                for (let d = 0; d < dimension; d++) {
                    gradient[i1][d] += term1[d];
                }
                
                // Term 2 & 3: Derivatives through the kernel
                for (const kd of kernelData) {
                    const { i, j, d_ij, d_norm, c_tilde, c_norm, K } = kd;
                    
                    // Derivative of |ẽ|^α through e_I (applies to all terms)
                    // d|ẽ|^α/d(γ_{i₁}) via e_I = α|ẽ|^(α-2) * (d × ẽ)  [times coeff/|d|^β]
                    // Note: d × (e × d) = |d|²e - (d·e)d
                    const d_cross_c = (dimension === 3) 
                        ? cross3D(d_ij, c_tilde)
                        : scale(c_tilde[2], subtract(scale(d_ij[0]*d_ij[0] + d_ij[1]*d_ij[1], [e_I[0]/ell_I, e_I[1]/ell_I]), 
                                                       scale(dot(d_ij, e_I)/ell_I, d_ij)));
                    
                    // Actually, let me compute this more carefully for 3D
                    // d × ẽ = d × (e × d) = |d|²e - (d·e)d (triple product identity)
                    let crossResult;
                    if (dimension === 3) {
                        crossResult = cross3D(d_ij, c_tilde);
                    } else {
                        // For 2D embedded in 3D
                        // ẽ = (0, 0, e₁d₂ - e₂d₁)
                        // d × ẽ requires d = (d₁, d₂, 0)
                        // This gives: (d₂ * ẽ_z, -d₁ * ẽ_z, 0) = ẽ_z * (d₂, -d₁, 0)
                        const c_z = c_tilde[2];
                        crossResult = [d_ij[1] * c_z, -d_ij[0] * c_z, 0];
                    }
                    
                    const term2_factor = alpha * Math.pow(c_norm, alpha - 2) / Math.pow(d_norm, beta);
                    const term2 = scale(coeff * term2_factor, crossResult);
                    
                    for (let d = 0; d < dimension; d++) {
                        gradient[i1][d] += term2[d];
                    }
                    
                    // Additional terms only when i = i1 (vertex p is the source point)
                    if (i === i1) {
                        // Derivative through d_{i₁j} in the cross product
                        // This adds: α|ẽ|^(α-2) * (e × ẽ) / |d|^β
                        // e × ẽ = e × (e × d) = (e·d)e - |e|²d
                        let e_cross_c;
                        if (dimension === 3) {
                            e_cross_c = cross3D(e_I, c_tilde);
                        } else {
                            const c_z = c_tilde[2];
                            e_cross_c = [e_I[1] * c_z, -e_I[0] * c_z, 0];
                        }
                        
                        const term3 = scale(coeff * term2_factor, e_cross_c);
                        
                        for (let d = 0; d < dimension; d++) {
                            gradient[i1][d] += term3[d];
                        }
                        
                        // Derivative of |d|^(-β)
                        // d|d|^(-β)/d(γ_{i₁}) = -β|d|^(-β-2) * d
                        const term4_factor = -beta * Math.pow(c_norm, alpha) * Math.pow(d_norm, -beta - 2);
                        const term4 = scale(coeff * term4_factor, d_ij);
                        
                        for (let d = 0; d < dimension; d++) {
                            gradient[i1][d] += term4[d];
                        }
                    }
                }
            }
            
            // ----- Vertex i2 (second endpoint of I) -----
            {
                // Term 1: Derivative of ℓ_I^(1-α)
                // d(ℓ_I^(1-α))/d(γ_{i₂}) = (1-α) * ℓ_I^(-α-1) * (γ_{i₂} - γ_{i₁})
                const term1_coeff = (1 - alpha) * Math.pow(ell_I, -alpha - 1) * sumK * 0.25 * ell_J;
                const term1 = scale(term1_coeff, subtract(vertices[i2], vertices[i1]));
                
                for (let d = 0; d < dimension; d++) {
                    gradient[i2][d] += term1[d];
                }
                
                // Term 2: Derivative through e_I (note: de_I/dγ_{i₂} = +I instead of -I)
                // This means the cross product term has opposite sign
                for (const kd of kernelData) {
                    const { i, j, d_ij, d_norm, c_tilde, c_norm, K } = kd;
                    
                    let crossResult;
                    if (dimension === 3) {
                        crossResult = cross3D(d_ij, c_tilde);
                    } else {
                        const c_z = c_tilde[2];
                        crossResult = [d_ij[1] * c_z, -d_ij[0] * c_z, 0];
                    }
                    
                    // Opposite sign compared to i1 case
                    const term2_factor = -alpha * Math.pow(c_norm, alpha - 2) / Math.pow(d_norm, beta);
                    const term2 = scale(coeff * term2_factor, crossResult);
                    
                    for (let d = 0; d < dimension; d++) {
                        gradient[i2][d] += term2[d];
                    }
                    
                    // Additional terms when i = i2
                    if (i === i2) {
                        // Derivative through d_{i₂j} in the cross product
                        // Note: d(d_{i₂j})/d(γ_{i₂}) = +I
                        let e_cross_c;
                        if (dimension === 3) {
                            e_cross_c = cross3D(e_I, c_tilde);
                        } else {
                            const c_z = c_tilde[2];
                            e_cross_c = [e_I[1] * c_z, -e_I[0] * c_z, 0];
                        }
                        
                        // Sign is positive here (d(d)/d(γ_{i₂}) = +I when differentiating d = γ_{i₂} - γ_j)
                        // Wait, d_{i₂j} = γ_{i₂} - γ_j, so d(d_{i₂j})/d(γ_{i₂}) = +I
                        // But the derivative formula gives -[e]_× for the cross product contribution
                        // Let me reconsider...
                        
                        // Actually for i = i₂: we have d = γ_{i₂} - γ_j
                        // The derivative of ẽ = e × d through d gives [e]_× * I = [e]_×
                        // So d|ẽ|^α/d(γ_{i₂})|_{via d} = α|ẽ|^(α-2) ẽᵀ [e]_× = α|ẽ|^(α-2) (e × ẽ)ᵀ
                        // But e × (e × d) = (e·d)e - |e|²d
                        
                        const term3_factor = alpha * Math.pow(c_norm, alpha - 2) / Math.pow(d_norm, beta);
                        const term3 = scale(coeff * term3_factor, e_cross_c);
                        
                        for (let d = 0; d < dimension; d++) {
                            gradient[i2][d] += term3[d];
                        }
                        
                        // Derivative of |d|^(-β) through d_{i₂j}
                        // d|d|^(-β)/d(γ_{i₂}) = -β|d|^(-β-2) * d (positive d because dd/dγ_{i₂} = +I)
                        const term4_factor = -beta * Math.pow(c_norm, alpha) * Math.pow(d_norm, -beta - 2);
                        const term4 = scale(coeff * term4_factor, d_ij);
                        
                        for (let d = 0; d < dimension; d++) {
                            gradient[i2][d] += term4[d];
                        }
                    }
                }
            }
            
            // ============================================================
            // CASE 2: Derivatives with respect to vertices of edge J
            // ============================================================
            
            // ----- Vertex j1 (first endpoint of J) -----
            {
                // Term 1: Derivative through ℓ_J
                // d(ℓ_J)/d(γ_{j₁}) = -T_J
                const term1_coeff = -0.25 * ell_I_pow * sumK;
                const term1 = scale(term1_coeff, T_J);
                
                for (let d = 0; d < dimension; d++) {
                    gradient[j1][d] += term1[d];
                }
                
                // Terms through the kernel (only when j = j1)
                for (const kd of kernelData) {
                    const { i, j, d_ij, d_norm, c_tilde, c_norm, K } = kd;
                    
                    if (j === j1) {
                        // Derivative through d_{ij₁} = γ_i - γ_{j₁}
                        // d(d)/d(γ_{j₁}) = -I
                        
                        // Cross product contribution: negative sign from chain rule
                        let e_cross_c;
                        if (dimension === 3) {
                            e_cross_c = cross3D(e_I, c_tilde);
                        } else {
                            const c_z = c_tilde[2];
                            e_cross_c = [e_I[1] * c_z, -e_I[0] * c_z, 0];
                        }
                        
                        // -α|ẽ|^(α-2) * (e × ẽ) * (-1) / |d|^β = +α|ẽ|^(α-2) * (e × ẽ) / |d|^β
                        // Wait, let me be more careful:
                        // d|ẽ|^α/d(γ_{j₁}) = d|ẽ|^α/d(d) * d(d)/d(γ_{j₁})
                        //                  = α|ẽ|^(α-2) ẽᵀ [e]_× * (-I)
                        //                  = -α|ẽ|^(α-2) (e × ẽ)ᵀ
                        const term2_factor = -alpha * Math.pow(c_norm, alpha - 2) / Math.pow(d_norm, beta);
                        const term2 = scale(coeff * term2_factor, e_cross_c);
                        
                        for (let d = 0; d < dimension; d++) {
                            gradient[j1][d] += term2[d];
                        }
                        
                        // Derivative of |d|^(-β)
                        // d|d|^(-β)/d(γ_{j₁}) = -β|d|^(-β-2) * d * (-1) = +β|d|^(-β-2) * d
                        const term3_factor = beta * Math.pow(c_norm, alpha) * Math.pow(d_norm, -beta - 2);
                        const term3 = scale(coeff * term3_factor, d_ij);
                        
                        for (let d = 0; d < dimension; d++) {
                            gradient[j1][d] += term3[d];
                        }
                    }
                }
            }
            
            // ----- Vertex j2 (second endpoint of J) -----
            {
                // Term 1: Derivative through ℓ_J
                // d(ℓ_J)/d(γ_{j₂}) = +T_J
                const term1_coeff = 0.25 * ell_I_pow * sumK;
                const term1 = scale(term1_coeff, T_J);
                
                for (let d = 0; d < dimension; d++) {
                    gradient[j2][d] += term1[d];
                }
                
                // Terms through the kernel (only when j = j2)
                for (const kd of kernelData) {
                    const { i, j, d_ij, d_norm, c_tilde, c_norm, K } = kd;
                    
                    if (j === j2) {
                        // Derivative through d_{ij₂} = γ_i - γ_{j₂}
                        // d(d)/d(γ_{j₂}) = -I (same as j1 case)
                        
                        let e_cross_c;
                        if (dimension === 3) {
                            e_cross_c = cross3D(e_I, c_tilde);
                        } else {
                            const c_z = c_tilde[2];
                            e_cross_c = [e_I[1] * c_z, -e_I[0] * c_z, 0];
                        }
                        
                        const term2_factor = -alpha * Math.pow(c_norm, alpha - 2) / Math.pow(d_norm, beta);
                        const term2 = scale(coeff * term2_factor, e_cross_c);
                        
                        for (let d = 0; d < dimension; d++) {
                            gradient[j2][d] += term2[d];
                        }
                        
                        const term3_factor = beta * Math.pow(c_norm, alpha) * Math.pow(d_norm, -beta - 2);
                        const term3 = scale(coeff * term3_factor, d_ij);
                        
                        for (let d = 0; d < dimension; d++) {
                            gradient[j2][d] += term3[d];
                        }
                    }
                }
            }
        }
    }
    
    return gradient;
}

/**
 * Simpler, cleaner implementation that more directly follows the paper's formulation.
 * This version is easier to verify against the derivation.
 */
export function calculateAnalyticalDifferentialSimple(vertices, edges, alpha, beta, disjointPairs, config) {
    const numVertices = vertices.length;
    const dimension = config?.dimension || 3;
    const epsilon = config?.epsilonKernel || 1e-10;
    
    // Initialize gradient
    const gradient = vertices.map(() => new Array(dimension).fill(0));
    
    // Helper to add vector to gradient
    const addToGradient = (vertexIdx, contribution) => {
        for (let d = 0; d < dimension; d++) {
            gradient[vertexIdx][d] += contribution[d] || 0;
        }
    };
    
    // Process each edge pair
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
            
            // Compute all kernel values
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
            
            // Common factor
            const ell_I_pow = Math.pow(ell_I, 1 - alpha);
            
            // ---- Derivatives w.r.t. i1 ----
            // Term from ℓ_I^(1-α)
            const dL_di1 = scale((1-alpha) * Math.pow(ell_I, -alpha-1) * sumK * 0.25 * ell_J, 
                                 subtract(vertices[i1], vertices[i2]));
            addToGradient(i1, dL_di1);
            
            // Terms from each kernel
            for (const {i, j, d, d_norm, c_norm} of kernelInfo) {
                // Derivative through e_I (affects all kernels)
                // The cross product d × (e × d) = |d|²e - (d·e)d
                const d_cross_ed = subtract(scale(d_norm*d_norm, e_I), 
                                            scale(dot(d, e_I), d));
                const factor = alpha * Math.pow(c_norm, alpha-2) / Math.pow(d_norm, beta);
                addToGradient(i1, scale(0.25 * ell_J * ell_I_pow * factor, d_cross_ed));
                
                // Additional terms when i = i1
                if (i === i1) {
                    // Through d in cross product: e × (e × d) = (e·d)e - |e|²d
                    const e_cross_ed = subtract(scale(dot(e_I, d), e_I), 
                                                scale(ell_I*ell_I, d));
                    addToGradient(i1, scale(0.25 * ell_J * ell_I_pow * factor, e_cross_ed));
                    
                    // Through |d|^(-β)
                    const d_factor = -beta * Math.pow(c_norm, alpha) / Math.pow(d_norm, beta+2);
                    addToGradient(i1, scale(0.25 * ell_J * ell_I_pow * d_factor, d));
                }
            }
            
            // ---- Derivatives w.r.t. i2 ----
            // Term from ℓ_I^(1-α)  
            const dL_di2 = scale((1-alpha) * Math.pow(ell_I, -alpha-1) * sumK * 0.25 * ell_J,
                                 subtract(vertices[i2], vertices[i1]));
            addToGradient(i2, dL_di2);
            
            // Terms from kernels (opposite sign for e_I derivative)
            for (const {i, j, d, d_norm, c_norm} of kernelInfo) {
                const d_cross_ed = subtract(scale(d_norm*d_norm, e_I), 
                                            scale(dot(d, e_I), d));
                const factor = alpha * Math.pow(c_norm, alpha-2) / Math.pow(d_norm, beta);
                // Note: opposite sign because de_I/dγ_{i2} = +I
                addToGradient(i2, scale(-0.25 * ell_J * ell_I_pow * factor, d_cross_ed));
                
                if (i === i2) {
                    const e_cross_ed = subtract(scale(dot(e_I, d), e_I),
                                                scale(ell_I*ell_I, d));
                    addToGradient(i2, scale(0.25 * ell_J * ell_I_pow * factor, e_cross_ed));
                    
                    const d_factor = -beta * Math.pow(c_norm, alpha) / Math.pow(d_norm, beta+2);
                    addToGradient(i2, scale(0.25 * ell_J * ell_I_pow * d_factor, d));
                }
            }
            
            // ---- Derivatives w.r.t. j1 ----
            // Through ℓ_J
            addToGradient(j1, scale(-0.25 * ell_I_pow * sumK, T_J));
            
            // Through kernels where j = j1
            for (const {i, j, d, d_norm, c_norm} of kernelInfo) {
                if (j !== j1) continue;
                
                const e_cross_ed = subtract(scale(dot(e_I, d), e_I),
                                            scale(ell_I*ell_I, d));
                const factor = alpha * Math.pow(c_norm, alpha-2) / Math.pow(d_norm, beta);
                // dd/dγ_{j1} = -I, so we get negative of the usual derivative
                addToGradient(j1, scale(-0.25 * ell_J * ell_I_pow * factor, e_cross_ed));
                
                const d_factor = -beta * Math.pow(c_norm, alpha) / Math.pow(d_norm, beta+2);
                // Negative from dd/dγ_{j1} = -I
                addToGradient(j1, scale(-0.25 * ell_J * ell_I_pow * d_factor, d));
            }
            
            // ---- Derivatives w.r.t. j2 ----
            // Through ℓ_J
            addToGradient(j2, scale(0.25 * ell_I_pow * sumK, T_J));
            
            // Through kernels where j = j2
            for (const {i, j, d, d_norm, c_norm} of kernelInfo) {
                if (j !== j2) continue;
                
                const e_cross_ed = subtract(scale(dot(e_I, d), e_I),
                                            scale(ell_I*ell_I, d));
                const factor = alpha * Math.pow(c_norm, alpha-2) / Math.pow(d_norm, beta);
                addToGradient(j2, scale(-0.25 * ell_J * ell_I_pow * factor, e_cross_ed));
                
                const d_factor = -beta * Math.pow(c_norm, alpha) / Math.pow(d_norm, beta+2);
                addToGradient(j2, scale(-0.25 * ell_J * ell_I_pow * d_factor, d));
            }
        }
    }
    
    return gradient;
}

export default { calculateAnalyticalDifferential, calculateAnalyticalDifferentialSimple };