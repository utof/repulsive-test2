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

// Project a direction vector from a point (for arrows)
// Returns screen-space direction that's consistent regardless of view
export function projectDirection(
    _origin: Vec3, // unused: kept for call-site symmetry; direction alone is rotated
    direction: Vec3,
    rotationY: number,
    rotationX: number,
): [number, number] {
    // Rotate the direction vector (not a point, so no translation)
    const rotated = rotate3D(direction, rotationY, rotationX);
    // For screen space: x goes right, y goes down (but we flip y in project)
    // So direction in screen space is (rotated.x, -rotated.y)
    const len = Math.sqrt(rotated[0] * rotated[0] + rotated[1] * rotated[1]);
    if (len < 1e-10) return [0, 0];
    return [rotated[0] / len, -rotated[1] / len];
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
