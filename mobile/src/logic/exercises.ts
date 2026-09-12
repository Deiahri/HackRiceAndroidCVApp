/**
 * Rep counting + form scoring per exercise. 1:1 port of prototype/exercises.py.
 *
 * Each exercise drives a small state machine off one "primary" joint angle
 * (knee for squats, elbow for push-ups) and collects form metrics while a rep is
 * in progress. When the rep completes it is scored 0-100 from weighted sub-scores.
 *
 *     waiting --(angle > up)--> top --(angle < up - hysteresis)--> descent
 *     descent --(angle < down)--> bottom --(angle > up)--> top   [rep counted]
 *     descent --(angle > up)--> top                              [partial rep, not counted]
 */
import {
  EMA, L_ANKLE, L_HIP, L_KNEE, R_ANKLE, R_HIP, R_KNEE, Pose, Side,
  angleFromVertical, dot, jointAngle,
} from './pose';

export type RepState = 'waiting' | 'top' | 'descent' | 'bottom';

const clip = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const round1 = (x: number) => Math.round(x * 10) / 10;

/** 100 at `good`, 0 at `bad`, linear in between (either direction). */
export function ramp(x: number, good: number, bad: number): number {
  const t = (x - good) / (bad - good);
  return 100 * (1 - clip(t, 0, 1));
}

export function mean(values: number[]): number {
  return values.reduce((s, v) => s + v, 0) / values.length;
}

/** numpy.percentile with the default "linear" interpolation. */
export function percentile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export type RepResult = {
  number: number;
  score: number;
  subscores: Record<string, number>;
  cues: string[];
  durationS: number;
};

type Evaluation = { subscores: Record<string, number>; cues: string[] };

export abstract class Exercise {
  abstract readonly name: string;
  abstract readonly label: string;
  /** Primary angle above this = top of the rep. */
  abstract readonly upThreshold: number;
  /** Primary angle below this = bottom reached, so the rep counts. */
  abstract readonly downThreshold: number;
  /** Must drop this far below upThreshold to start a rep. */
  readonly hysteresis: number = 10;
  /** Dropped this far but never reached bottom = partial rep. */
  readonly partialMargin: number = 20;
  readonly minRepSeconds: number = 1.0;
  abstract readonly weights: Record<string, number>;

  state: RepState = 'waiting';
  reps: RepResult[] = [];
  partialReps = 0;
  /** Smoothed primary angle. */
  angle: number | null = null;
  /** Blocking problem, e.g. body not visible. */
  status = '';
  /** Sticky message, e.g. partial rep. */
  notice = '';

  protected side!: Side;
  protected minAngle = 180;
  protected lockoutAngle = 0;
  protected metrics: Record<string, number[]> = {};
  private ema = new EMA(0.5);
  private topPeak = 0;
  private tStart = 0;

  constructor() {
    this.startRep(0);
  }

  // --- per-exercise hooks -------------------------------------------------
  /** Angle that drives the state machine. Return null (optionally setting this.status) if unusable. */
  protected abstract primaryAngle(pose: Pose): number | null;

  /** Record form metrics for the current frame while a rep is in progress. */
  protected collect(_pose: Pose): void {}

  /** Sub-scores (0-100) and coaching cues for the finished rep. */
  protected abstract evaluate(): Evaluation;

  // --- state machine ------------------------------------------------------
  get avgScore(): number {
    return this.reps.length ? mean(this.reps.map((r) => r.score)) : 0;
  }

  protected record(key: string, value: number): void {
    (this.metrics[key] ??= []).push(value);
  }

  private startRep(t: number): void {
    this.tStart = t;
    this.minAngle = 180;
    this.lockoutAngle = this.topPeak; // how straight the joint got at the top before this rep
    this.metrics = {};
  }

  update(pose: Pose | null, t: number): RepResult | null {
    this.status = 'Body not fully visible';
    const raw = pose !== null ? this.primaryAngle(pose) : null;
    if (raw === null || pose === null) return null;
    this.status = '';
    const a = (this.angle = this.ema.update(raw));

    if (this.state === 'waiting') {
      if (a > this.upThreshold) {
        this.state = 'top';
        this.topPeak = a;
      } else {
        this.status = 'Get into the starting position';
      }
      return null;
    }

    if (this.state === 'top') {
      this.topPeak = Math.max(this.topPeak, a);
      if (a >= this.upThreshold - this.hysteresis) return null;
      this.startRep(t);
      this.state = 'descent';
    }

    // Rep in progress (descent or bottom)
    this.minAngle = Math.min(this.minAngle, a);
    this.collect(pose);
    if (this.state === 'descent' && a < this.downThreshold) {
      this.state = 'bottom';
    } else if (a > this.upThreshold) {
      const reachedBottom = this.state === 'bottom';
      this.state = 'top';
      this.topPeak = a;
      if (reachedBottom) return this.finishRep(t);
      if (this.minAngle < this.upThreshold - this.partialMargin) {
        this.partialReps += 1;
        this.notice = 'Partial rep - not counted, go deeper';
      }
    }
    return null;
  }

  private finishRep(t: number): RepResult {
    const { subscores, cues } = this.evaluate();
    const keys = Object.keys(subscores);
    const totalW = keys.reduce((s, k) => s + this.weights[k], 0);
    const score = keys.reduce((s, k) => s + subscores[k] * this.weights[k], 0) / totalW;
    const duration = t - this.tStart;
    if (duration < this.minRepSeconds) cues.push('Slow down');
    const rounded: Record<string, number> = {};
    for (const k of keys) rounded[k] = round1(subscores[k]);
    const rep: RepResult = {
      number: this.reps.length + 1,
      score: round1(score),
      subscores: rounded,
      cues: cues.length ? cues : ['Good rep'],
      durationS: Math.round(duration * 100) / 100,
    };
    this.reps.push(rep);
    this.notice = '';
    return rep;
  }
}

export class Squat extends Exercise {
  readonly name = 'squat';
  readonly label = 'Squat';
  readonly upThreshold = 160;
  readonly downThreshold = 110;
  readonly weights = { depth: 0.5, torso: 0.3, symmetry: 0.2 };
  readonly minRepSeconds = 1.0;

  protected primaryAngle(pose: Pose): number | null {
    const side = pose.bestSide('hip', 'knee', 'ankle');
    if (side === null) return null;
    this.side = side;
    return jointAngle(pose.pts[side.hip], pose.pts[side.knee], pose.pts[side.ankle]);
  }

  protected collect(pose: Pose): void {
    const s = this.side;
    if (pose.visible(s.shoulder)) {
      this.record('torso_lean', angleFromVertical(pose.pts[s.shoulder], pose.pts[s.hip]));
    }
    if (pose.visible(L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE)) {
      const left = jointAngle(pose.pts[L_HIP], pose.pts[L_KNEE], pose.pts[L_ANKLE]);
      const right = jointAngle(pose.pts[R_HIP], pose.pts[R_KNEE], pose.pts[R_ANKLE]);
      this.record('knee_diff', Math.abs(left - right));
    }
  }

  protected evaluate(): Evaluation {
    // 100% at <=90 deg knee (thighs parallel), 50% at 110
    const subscores: Record<string, number> = { depth: ramp(this.minAngle, 90, 130) };
    const cues: string[] = [];
    if (this.minAngle > 100) cues.push('Go deeper - aim for thighs parallel');
    const lean = this.metrics.torso_lean;
    if (lean?.length) {
      subscores.torso = ramp(percentile(lean, 90), 45, 75);
      if (subscores.torso < 70) cues.push('Keep your chest up');
    }
    const diff = this.metrics.knee_diff;
    if (diff && diff.length >= 3) {
      subscores.symmetry = ramp(mean(diff), 10, 35);
      if (subscores.symmetry < 70) cues.push('Uneven legs - balance your weight');
    }
    return { subscores, cues };
  }
}

export class PushUp extends Exercise {
  readonly name = 'pushup';
  readonly label = 'Push-up';
  readonly upThreshold = 150;
  readonly downThreshold = 100;
  readonly weights = { depth: 0.4, body_line: 0.4, lockout: 0.2 };
  readonly minRepSeconds = 0.8;

  protected primaryAngle(pose: Pose): number | null {
    const side = pose.bestSide('shoulder', 'elbow', 'wrist', 'hip', 'ankle');
    if (side === null) return null;
    // Only track when the body is roughly horizontal so arm bends while standing don't count
    if (angleFromVertical(pose.pts[side.shoulder], pose.pts[side.ankle]) < 45) {
      this.status = 'Get into a plank (side-on to camera)';
      return null;
    }
    this.side = side;
    return jointAngle(pose.pts[side.shoulder], pose.pts[side.elbow], pose.pts[side.wrist]);
  }

  protected collect(pose: Pose): void {
    const s = this.side;
    const sh = pose.pts[s.shoulder], hip = pose.pts[s.hip], ank = pose.pts[s.ankle];
    const deviation = 180 - jointAngle(sh, hip, ank); // 0 = straight shoulder-hip-ankle line
    // Where the hip sits relative to the shoulder->ankle line: below (larger y) = sagging
    const hipRel = hip.map((v, i) => v - sh[i]);
    const line = ank.map((v, i) => v - sh[i]);
    const t = dot(hipRel, line) / (dot(line, line) + 1e-9);
    const sagging = hip[1] > sh[1] + t * line[1];
    this.record('line_dev', sagging ? deviation : -deviation);
  }

  protected evaluate(): Evaluation {
    const devs = this.metrics.line_dev ?? [0];
    const subscores: Record<string, number> = {
      depth: ramp(this.minAngle, 90, 130), // 100% at <=90 deg elbow
      body_line: ramp(percentile(devs.map(Math.abs), 90), 10, 30),
      lockout: ramp(this.lockoutAngle, 165, 140), // arms straight at the top
    };
    const cues: string[] = [];
    if (this.minAngle > 95) cues.push('Lower your chest further');
    if (subscores.body_line < 70) {
      cues.push(mean(devs) > 0 ? 'Hips sagging - brace your core' : 'Hips too high - flatten your body');
    }
    if (subscores.lockout < 70) cues.push('Fully extend your arms at the top');
    return { subscores, cues };
  }
}

export type ExerciseName = 'squat' | 'pushup';

export const EXERCISES: Record<ExerciseName, () => Exercise> = {
  squat: () => new Squat(),
  pushup: () => new PushUp(),
};
