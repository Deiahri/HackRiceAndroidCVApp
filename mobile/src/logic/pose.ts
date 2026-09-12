/**
 * BlazePose landmark indices + geometry helpers.
 * 1:1 port of prototype/pose.py (everything below PoseEstimator there).
 */

export const VIS_THRESHOLD = 0.5;

// BlazePose landmark indices (33 total). "Left" is the person's left.
export const L_SHOULDER = 11, R_SHOULDER = 12;
export const L_ELBOW = 13, R_ELBOW = 14;
export const L_WRIST = 15, R_WRIST = 16;
export const L_HIP = 23, R_HIP = 24;
export const L_KNEE = 25, R_KNEE = 26;
export const L_ANKLE = 27, R_ANKLE = 28;

export type Side = { shoulder: number; elbow: number; wrist: number; hip: number; knee: number; ankle: number };
export type Joint = keyof Side;

export const SIDES: { left: Side; right: Side } = {
  left: { shoulder: L_SHOULDER, elbow: L_ELBOW, wrist: L_WRIST, hip: L_HIP, knee: L_KNEE, ankle: L_ANKLE },
  right: { shoulder: R_SHOULDER, elbow: R_ELBOW, wrist: R_WRIST, hip: R_HIP, knee: R_KNEE, ankle: R_ANKLE },
};

export const SKELETON: [number, number][] = [
  [L_SHOULDER, R_SHOULDER], [L_HIP, R_HIP],
  [L_SHOULDER, L_ELBOW], [L_ELBOW, L_WRIST], [R_SHOULDER, R_ELBOW], [R_ELBOW, R_WRIST],
  [L_SHOULDER, L_HIP], [R_SHOULDER, R_HIP],
  [L_HIP, L_KNEE], [L_KNEE, L_ANKLE], [R_HIP, R_KNEE], [R_KNEE, R_ANKLE],
];

/** A 2D ([x, y]) or 3D ([x, y, z]) point. */
export type Vec = readonly number[];

export class Pose {
  /** Points the angles are computed on: pixel coords (2D mode) or world coords in metres (3D mode). */
  pts: Vec[];
  /** Pixel coords, for drawing. */
  px: Vec[];
  /** Visibility 0..1 per landmark. */
  vis: number[];

  constructor(pts: Vec[], px: Vec[], vis: number[]) {
    this.pts = pts;
    this.px = px;
    this.vis = vis;
  }

  visible(...idx: number[]): boolean {
    return idx.every((i) => this.vis[i] >= VIS_THRESHOLD);
  }

  /** Joint map of the side whose `joints` are most visible, or null if neither side is usable. */
  bestSide(...joints: Joint[]): Side | null {
    const score = (s: Side) => joints.reduce((sum, j) => sum + this.vis[s[j]], 0);
    // Python's max() keeps the first of equal keys, so left wins ties.
    const side = score(SIDES.right) > score(SIDES.left) ? SIDES.right : SIDES.left;
    return this.visible(...joints.map((j) => side[j])) ? side : null;
  }
}

type NormalizedLandmark = { x: number; y: number; z: number; visibility: number };
type WorldLandmark = { x: number; y: number; z: number };

/**
 * Build a Pose from PoseLandmarker output (normalized image coords), like PoseEstimator.detect().
 * Returns null when no person was detected.
 */
export function poseFromLandmarks(
  landmarks: NormalizedLandmark[],
  imageWidth: number,
  imageHeight: number,
  world?: WorldLandmark[],
  use3d = false,
): Pose | null {
  if (landmarks.length === 0) return null;
  const px = landmarks.map((lm) => [lm.x * imageWidth, lm.y * imageHeight]);
  const vis = landmarks.map((lm) => lm.visibility ?? 0);
  const pts = use3d && world && world.length ? world.map((lm) => [lm.x, lm.y, lm.z]) : px;
  return new Pose(pts, px, vis);
}

const clip = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const degrees = (rad: number) => (rad * 180) / Math.PI;
const sub = (a: Vec, b: Vec) => a.map((v, i) => v - b[i]);
export const dot = (a: Vec, b: Vec) => a.reduce((s, v, i) => s + v * b[i], 0);
const norm = (a: Vec) => Math.sqrt(dot(a, a));

/** Angle ABC in degrees, at vertex b. Works for 2D or 3D points. */
export function jointAngle(a: Vec, b: Vec, c: Vec): number {
  const ba = sub(a, b), bc = sub(c, b);
  const cos = dot(ba, bc) / (norm(ba) * norm(bc) + 1e-9);
  return degrees(Math.acos(clip(cos, -1, 1)));
}

/** Angle in degrees between segment bottom->top and straight up (0 = upright, 90 = horizontal). */
export function angleFromVertical(top: Vec, bottom: Vec): number {
  const v = sub(top, bottom);
  const cos = -v[1] / (norm(v) + 1e-9); // "up" is -y in both image and MediaPipe world coords
  return degrees(Math.acos(clip(cos, -1, 1)));
}

/** Exponential moving average to damp keypoint jitter. */
export class EMA {
  alpha: number;
  value: number | null = null;

  constructor(alpha = 0.5) {
    this.alpha = alpha;
  }

  update(x: number): number {
    this.value = this.value === null ? x : this.alpha * x + (1 - this.alpha) * this.value;
    return this.value;
  }
}
