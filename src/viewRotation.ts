import type { Vec3 } from './testConfigs';

// 3D rotation (returns rotated point).
// Orbit model: rotationY = azimuth (yaw about world +Y), rotationX = pitch. This
// is the same up-locked turntable model three.js OrbitControls uses.
function rotate3D(point: Vec3, rotationY: number, rotationX: number): Vec3 {
    const [x, y, z] = point;

    // Rotate around Y
    const cosY = Math.cos(rotationY);
    const sinY = Math.sin(rotationY);
    const x1 = x * cosY - z * sinY;
    const z1 = x * sinY + z * cosY;

    // Rotate around X
    const cosX = Math.cos(rotationX);
    const sinX = Math.sin(rotationX);
    const y1 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX;

    return [x1, y1, z2];
}

// 3D projection
export function project(
    point: Vec3,
    width: number,
    height: number,
    rotationY: number,
    rotationX: number,
    zoom: number = 1,
): [number, number] {
    const [x1, y1, z2] = rotate3D(point, rotationY, rotationX);

    // Simple perspective
    const fov = 3;
    const s = fov / (fov + z2 + 3);
    const screenX = width / 2 + x1 * s * 150 * zoom;
    const screenY = height / 2 - y1 * s * 150 * zoom;

    return [screenX, screenY];
}

// Project a gradient arrow (negative-gradient direction) to screen space.
//
// Direction is the ORTHOGRAPHIC projection of the rotated vector (its in-plane part);
// magnitude is a separate log-compressed, clamped scalar, foreshortened by how much of
// the vector lies in the screen plane. WHY this shape — each part guards a real,
// confirmed bug; do not "simplify":
//
//   1. Direction from the orthographic in-plane part (dirX, -dirY) of rotate3D(negGrad),
//      NOT the full local projection derivative. The perspective derivative carries a
//      position-dependent term (X·ds, Y·ds) that, for a gradient pointing near the view
//      axis, OVERTAKES the true in-plane lean and flips sign as the vertex crosses the
//      canvas centre — the arrow whipped 180° for a smooth orbit (the "changes direction
//      dramatically near the centre" bug). Orthographic direction depends only on the
//      vector's orientation, never on screen position, so it rotates smoothly. The one
//      remaining reversal is the TRUE degeneracy (vector aimed exactly at the camera),
//      where inPlane -> 0 and the arrow honestly shrinks to a dot instead of whipping.
//   2. Only the BASE vertex is projected — never a far tip project(vertex + k·negGrad).
//      Near-intersection gradients are astronomically large; a projected far tip reached
//      the perspective near-plane (s = 3/(z2+6)) where s exploded / flipped sign and the
//      arrow shot off-canvas. The base is a real graph vertex, always well-conditioned.
//   3. minLen/maxLen clamp the MAGNITUDE before foreshortening, not the final length:
//      minLen keeps a small-|grad| arrow visible; maxLen caps huge gradients; and because
//      the clamp precedes the inPlane foreshorten, an arrow aimed at the camera still
//      collapses to ~0 (no floored stub whipping at the degeneracy).
export function projectArrow(
    vertex: Vec3,
    negGrad: Vec3,
    gradNorm: number,
    width: number,
    height: number,
    rotationY: number,
    rotationX: number,
    zoom: number,
    scale: number,
    minLen: number,
    maxLen: number,
): { baseX: number; baseY: number; tipX: number; tipY: number; len: number } {
    // Base + per-world screen scale k, mirroring project() exactly so arrow bases sit on
    // their vertices (same fov, same s, same op order). @see project above.
    const [X, Y, Z] = rotate3D(vertex, rotationY, rotationX);
    const fov = 3;
    const s = fov / (fov + Z + 3);
    const k = s * 150 * zoom; // screen px per world-unit for an in-plane step
    const baseX = width / 2 + X * s * 150 * zoom;
    const baseY = height / 2 - Y * s * 150 * zoom;

    // Orthographic in-plane direction of the rotated vector (screen y grows downward).
    const [gx, gy] = rotate3D(
        [negGrad[0] / gradNorm, negGrad[1] / gradNorm, negGrad[2] / gradNorm],
        rotationY,
        rotationX,
    );
    const dirX = gx;
    const dirY = -gy;
    const inPlane = Math.hypot(dirX, dirY); // fraction of the unit vector in the screen plane

    // A vector pointing straight at/away from the camera has no on-screen direction.
    if (inPlane < 1e-12) return { baseX, baseY, tipX: baseX, tipY: baseY, len: 0 };

    // Log-compressed world length -> full-arrow pixels, clamped BEFORE foreshortening.
    const worldLen = Math.log(1 + gradNorm) * scale;
    const magPx = Math.max(minLen, Math.min(maxLen, k * worldLen));

    // Tip = base + unit-direction · magPx; foreshortened drawn length is magPx·inPlane.
    return {
        baseX,
        baseY,
        tipX: baseX + dirX * magPx,
        tipY: baseY + dirY * magPx,
        len: magPx * inPlane,
    };
}

// --- View-orbit pitch clamp -------------------------------------------------
// WHY THIS EXISTS (do not remove — it fixes a real, confirmed bug):
// This viewer orbits with two world-frame angles (rotationY = azimuth about world
// +Y, rotationX = pitch) — the same up-locked turntable model three.js
// OrbitControls uses. OrbitControls, however, clamps its polar angle to [0, PI]
// (minPolarAngle=0 / maxPolarAngle=PI defaults) so the camera can never tip over
// the pole into an upside-down view. This code originally had NO such clamp.
//
// Without the clamp, once the pitch passes the pole the scene is upside-down and a
// "drag up" starts moving the scene DOWN — the vertical drag silently reverses,
// and re-grabbing can't fix it because the reversal is a pure function of the
// persistent pitch, not of any per-drag state. That is exactly the reported
// "dragging in 3D is unintuitive / doesn't reset" symptom. Clamping the pitch
// makes the upside-down regime unreachable, so the reversal can never happen.
//
// Mapping to OrbitControls' Spherical(radius, phi, theta) (phi measured from +Y):
// here rotationX = 0 is the equator (phi = PI/2), so phi = PI/2 - rotationX, and
// OrbitControls' phi in [0, PI]  <=>  rotationX in [-PI/2, PI/2] = [-POLAR_LIMIT, POLAR_LIMIT].
//
// MIGRATION NOTE (react-three-fiber): delete this module and hand the same limits
// to <OrbitControls minPolarAngle={0} maxPolarAngle={Math.PI} /> (its defaults),
// which reproduces this behavior — do NOT drop the pitch clamp in the process.
// @see https://threejs.org/docs/#examples/en/controls/OrbitControls (.min/.maxPolarAngle)
export const POLAR_LIMIT = Math.PI / 2;

export function clampPolar(rotationX: number): number {
    return Math.max(-POLAR_LIMIT, Math.min(POLAR_LIMIT, rotationX));
}
