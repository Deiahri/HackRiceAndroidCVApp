/**
 * Replays synthetic side-view poses through the rep/score engine (same scenarios as Findings §3a).
 *
 *     npm run test:logic            # run the checks
 *     npm run test:logic -- --dump  # print the frames as JSON (to cross-check against prototype/exercises.py)
 */
import { EXERCISES, Exercise, ExerciseName } from './exercises';
import { Pose, SIDES, Vec } from './pose';

const FPS = 30;
const rad = (deg: number) => (deg * Math.PI) / 180;

function blankPose(): { pts: number[][]; vis: number[] } {
  return { pts: Array.from({ length: 33 }, () => [0, 0]), vis: Array(33).fill(1) };
}

/** Side view: shin vertical, thigh at `knee` degrees, torso leaning `lean` degrees from vertical. */
function squatPose(knee: number, lean: number): Pose {
  const { pts, vis } = blankPose();
  for (const s of [SIDES.left, SIDES.right]) {
    const kneeP = [300, 300];
    const hip = [kneeP[0] - 100 * Math.sin(rad(knee)), kneeP[1] + 100 * Math.cos(rad(knee))];
    pts[s.ankle] = [300, 400];
    pts[s.knee] = kneeP;
    pts[s.hip] = hip;
    pts[s.shoulder] = [hip[0] + 150 * Math.sin(rad(lean)), hip[1] - 150 * Math.cos(rad(lean))];
  }
  return new Pose(pts, pts, vis);
}

/** Side-view plank: forearm vertical, upper arm at `elbow` degrees, hip dropped `sag` px below the line. */
function pushupPose(elbow: number, sag: number): Pose {
  const { pts, vis } = blankPose();
  for (const s of [SIDES.left, SIDES.right]) {
    const elbowP = [200, 340];
    const sh = [elbowP[0] - 60 * Math.sin(rad(elbow)), elbowP[1] + 60 * Math.cos(rad(elbow))];
    pts[s.wrist] = [200, 400];
    pts[s.elbow] = elbowP;
    pts[s.shoulder] = sh;
    pts[s.hip] = [sh[0] + 150, sh[1] + sag];
    pts[s.knee] = [sh[0] + 225, sh[1] + sag / 2];
    pts[s.ankle] = [sh[0] + 300, sh[1]];
  }
  return new Pose(pts, pts, vis);
}

/** Primary-angle trajectory: rest at the top, ease down to `bottom`, hold, ease back up; `reps` times. */
function trajectory(bottom: number, reps: number, top = 175): number[] {
  const angles: number[] = Array(15).fill(top);
  for (let r = 0; r < reps; r++) {
    for (let i = 1; i <= 20; i++) angles.push(top + (bottom - top) * (1 - Math.cos((Math.PI * i) / 20)) / 2);
    for (let i = 0; i < 10; i++) angles.push(bottom);
    for (let i = 1; i <= 20; i++) angles.push(bottom + (top - bottom) * (1 - Math.cos((Math.PI * i) / 20)) / 2);
    for (let i = 0; i < 15; i++) angles.push(top);
  }
  return angles;
}

type Scenario = {
  name: string;
  exercise: ExerciseName;
  frames: (Pose | null)[];
  check: (ex: Exercise) => string | null; // null = pass, otherwise the failure reason
};

const squat = (bottom: number, reps: number, lean = 20) => trajectory(bottom, reps).map((a) => squatPose(a, lean));
const pushup = (bottom: number, reps: number, sag = 0) => trajectory(bottom, reps).map((a) => pushupPose(a, sag));

const expect = (cond: boolean, why: string) => (cond ? null : why);
const scores = (ex: Exercise) => ex.reps.map((r) => r.score);
const cues = (ex: Exercise) => ex.reps.flatMap((r) => r.cues).join(' | ');
const near = (x: number, target: number, tol: number) => Math.abs(x - target) <= tol;

const SCENARIOS: Scenario[] = [
  {
    name: 'Squat: 3 good reps', exercise: 'squat', frames: squat(85, 3),
    check: (ex) => expect(ex.reps.length === 3 && scores(ex).every((s) => s === 100), 'expected 3 x 100%'),
  },
  {
    name: 'Squat: shallow (135°)', exercise: 'squat', frames: squat(135, 1),
    check: (ex) => expect(ex.reps.length === 0 && ex.partialReps === 1, 'expected 0 reps, 1 partial'),
  },
  {
    name: 'Squat: 105° depth', exercise: 'squat', frames: squat(105, 1),
    check: (ex) => expect(ex.reps.length === 1 && near(ex.reps[0].score, 80, 3) && cues(ex).includes('Go deeper'),
      'expected 1 rep ~80% with "Go deeper"'),
  },
  {
    name: 'Squat: 65° torso lean', exercise: 'squat', frames: squat(85, 1, 65),
    check: (ex) => expect(ex.reps.length === 1 && near(ex.reps[0].score, 80, 1) && cues(ex).includes('Keep your chest up'),
      'expected 1 rep ~80% with "Keep your chest up"'),
  },
  {
    name: 'Push-up: 3 good reps', exercise: 'pushup', frames: pushup(85, 3),
    check: (ex) => expect(ex.reps.length === 3 && scores(ex).every((s) => s === 100), 'expected 3 x 100%'),
  },
  {
    name: 'Push-up: hips sagging', exercise: 'pushup', frames: pushup(85, 1, 45),
    check: (ex) => expect(ex.reps.length === 1 && near(ex.reps[0].score, 60, 1) && cues(ex).includes('Hips sagging'),
      'expected 1 rep ~60% with "Hips sagging"'),
  },
  {
    name: 'Push-up: shallow (115°)', exercise: 'pushup', frames: pushup(115, 1),
    check: (ex) => expect(ex.reps.length === 0 && ex.partialReps === 1, 'expected 0 reps, 1 partial'),
  },
  {
    name: 'No person in frame', exercise: 'squat', frames: Array(10).fill(null),
    check: (ex) => expect(ex.reps.length === 0 && ex.status === 'Body not fully visible', 'expected "Body not fully visible"'),
  },
  {
    name: 'Push-up: standing up', exercise: 'pushup', frames: squat(175, 0),
    check: (ex) => expect(ex.state === 'waiting' && ex.status.startsWith('Get into a plank'), 'expected "Get into a plank"'),
  },
];

function run(s: Scenario): Exercise {
  const ex = EXERCISES[s.exercise]();
  s.frames.forEach((pose, i) => ex.update(pose, i / FPS));
  return ex;
}

const proc = (globalThis as unknown as { process?: { argv: string[]; exitCode?: number } }).process;

if (proc?.argv.includes('--dump')) {
  const round = (p: Vec) => p.map((v) => Math.round(v * 1000) / 1000);
  const out = SCENARIOS.map((s) => {
    const ex = run(s);
    return {
      name: s.name,
      exercise: s.exercise,
      frames: s.frames.map((p) => (p ? { pts: p.pts.map(round), vis: p.vis } : null)),
      ts: { reps: ex.reps, partialReps: ex.partialReps, status: ex.status, state: ex.state },
    };
  });
  console.log(JSON.stringify(out));
} else {
  let failed = 0;
  for (const s of SCENARIOS) {
    const ex = run(s);
    const err = s.check(ex);
    failed += err ? 1 : 0;
    const summary = `reps=${ex.reps.length} partial=${ex.partialReps} scores=[${scores(ex).join(', ')}]`
      + (ex.reps.length ? ` cues="${cues(ex)}"` : '') + (ex.status ? ` status="${ex.status}"` : '');
    console.log(`${err ? 'FAIL' : 'PASS'}  ${s.name.padEnd(26)} ${summary}${err ? `  <- ${err}` : ''}`);
  }
  console.log(`\n${SCENARIOS.length - failed}/${SCENARIOS.length} passed`);
  if (proc && failed) proc.exitCode = 1;
}
