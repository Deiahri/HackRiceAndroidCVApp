"""Rep counting + form scoring per exercise.

Each exercise drives a small state machine off one "primary" joint angle
(knee for squats, elbow for push-ups) and collects form metrics while a rep is
in progress. When the rep completes it is scored 0-100 from weighted
sub-scores. Pure geometry, no training data, so it ports directly to TypeScript.

    waiting --(angle > up)--> top --(angle < up - hysteresis)--> descent
    descent --(angle < down)--> bottom --(angle > up)--> top   [rep counted]
    descent --(angle > up)--> top                              [partial rep, not counted]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pose import (EMA, L_ANKLE, L_HIP, L_KNEE, R_ANKLE, R_HIP, R_KNEE, Pose,
                  angle_from_vertical, joint_angle)

WAITING, TOP, DESCENT, BOTTOM = "waiting", "top", "descent", "bottom"


def ramp(x: float, good: float, bad: float) -> float:
    """100 at `good`, 0 at `bad`, linear in between (either direction)."""
    t = (x - good) / (bad - good)
    return float(100 * (1 - np.clip(t, 0, 1)))


@dataclass
class RepResult:
    number: int
    score: float
    subscores: dict[str, float]
    cues: list[str]
    duration_s: float


class Exercise:
    name: str
    up_threshold: float     # primary angle above this = top of the rep
    down_threshold: float   # primary angle below this = bottom reached, so the rep counts
    hysteresis = 10.0       # must drop this far below up_threshold to start a rep
    partial_margin = 20.0   # dropped this far but never reached bottom = partial rep
    min_rep_seconds = 1.0
    weights: dict[str, float]

    def __init__(self):
        self.state = WAITING
        self.reps: list[RepResult] = []
        self.partial_reps = 0
        self.angle: float | None = None  # smoothed primary angle
        self.status = ""                 # blocking problem, e.g. body not visible
        self.notice = ""                 # sticky message, e.g. partial rep
        self._ema = EMA(0.5)
        self._top_peak = 0.0
        self._start_rep(0.0)

    # --- per-exercise hooks -------------------------------------------------
    def primary_angle(self, pose: Pose) -> float | None:
        """Angle that drives the state machine. Return None (optionally setting self.status) if unusable."""
        raise NotImplementedError

    def collect(self, pose: Pose) -> None:
        """Record form metrics for the current frame while a rep is in progress."""

    def evaluate(self) -> tuple[dict[str, float], list[str]]:
        """Sub-scores (0-100) and coaching cues for the finished rep."""
        raise NotImplementedError

    # --- state machine ------------------------------------------------------
    @property
    def avg_score(self) -> float:
        return float(np.mean([r.score for r in self.reps])) if self.reps else 0.0

    def record(self, key: str, value: float) -> None:
        self.metrics.setdefault(key, []).append(value)

    def _start_rep(self, t: float) -> None:
        self.t_start = t
        self.min_angle = 180.0
        self.lockout_angle = self._top_peak  # how straight the joint got at the top before this rep
        self.metrics: dict[str, list[float]] = {}

    def update(self, pose: Pose | None, t: float) -> RepResult | None:
        self.status = "Body not fully visible"
        raw = self.primary_angle(pose) if pose is not None else None
        if raw is None:
            return None
        self.status = ""
        a = self.angle = self._ema.update(raw)

        if self.state == WAITING:
            if a > self.up_threshold:
                self.state, self._top_peak = TOP, a
            else:
                self.status = "Get into the starting position"
            return None

        if self.state == TOP:
            self._top_peak = max(self._top_peak, a)
            if a >= self.up_threshold - self.hysteresis:
                return None
            self._start_rep(t)
            self.state = DESCENT

        # Rep in progress (DESCENT or BOTTOM)
        self.min_angle = min(self.min_angle, a)
        self.collect(pose)
        if self.state == DESCENT and a < self.down_threshold:
            self.state = BOTTOM
        elif a > self.up_threshold:
            reached_bottom = self.state == BOTTOM
            self.state, self._top_peak = TOP, a
            if reached_bottom:
                return self._finish_rep(t)
            if self.min_angle < self.up_threshold - self.partial_margin:
                self.partial_reps += 1
                self.notice = "Partial rep - not counted, go deeper"
        return None

    def _finish_rep(self, t: float) -> RepResult:
        subscores, cues = self.evaluate()
        total_w = sum(self.weights[k] for k in subscores)
        score = sum(v * self.weights[k] for k, v in subscores.items()) / total_w
        duration = t - self.t_start
        if duration < self.min_rep_seconds:
            cues.append("Slow down")
        rep = RepResult(len(self.reps) + 1, round(score, 1), {k: round(v, 1) for k, v in subscores.items()},
                        cues or ["Good rep"], round(duration, 2))
        self.reps.append(rep)
        self.notice = ""
        return rep


class Squat(Exercise):
    name = "squat"
    up_threshold = 160.0
    down_threshold = 110.0
    weights = {"depth": 0.5, "torso": 0.3, "symmetry": 0.2}
    min_rep_seconds = 1.0

    def primary_angle(self, pose):
        side = pose.best_side("hip", "knee", "ankle")
        if side is None:
            return None
        self._side = side
        return joint_angle(pose.pts[side["hip"]], pose.pts[side["knee"]], pose.pts[side["ankle"]])

    def collect(self, pose):
        s = self._side
        if pose.visible(s["shoulder"]):
            self.record("torso_lean", angle_from_vertical(pose.pts[s["shoulder"]], pose.pts[s["hip"]]))
        if pose.visible(L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE):
            left = joint_angle(pose.pts[L_HIP], pose.pts[L_KNEE], pose.pts[L_ANKLE])
            right = joint_angle(pose.pts[R_HIP], pose.pts[R_KNEE], pose.pts[R_ANKLE])
            self.record("knee_diff", abs(left - right))

    def evaluate(self):
        subs = {"depth": ramp(self.min_angle, 90, 130)}  # 100% at <=90 deg knee (thighs parallel), 50% at 110
        cues = []
        if self.min_angle > 100:
            cues.append("Go deeper - aim for thighs parallel")
        if lean := self.metrics.get("torso_lean"):
            subs["torso"] = ramp(np.percentile(lean, 90), 45, 75)
            if subs["torso"] < 70:
                cues.append("Keep your chest up")
        if (diff := self.metrics.get("knee_diff")) and len(diff) >= 3:
            subs["symmetry"] = ramp(np.mean(diff), 10, 35)
            if subs["symmetry"] < 70:
                cues.append("Uneven legs - balance your weight")
        return subs, cues


class PushUp(Exercise):
    name = "pushup"
    up_threshold = 150.0
    down_threshold = 100.0
    weights = {"depth": 0.4, "body_line": 0.4, "lockout": 0.2}
    min_rep_seconds = 0.8
    _joints = ("shoulder", "elbow", "wrist", "hip", "ankle")

    def primary_angle(self, pose):
        side = pose.best_side(*self._joints)
        if side is None:
            return None
        # Only track when the body is roughly horizontal so arm bends while standing don't count
        if angle_from_vertical(pose.pts[side["shoulder"]], pose.pts[side["ankle"]]) < 45:
            self.status = "Get into a plank (side-on to camera)"
            return None
        self._side = side
        return joint_angle(pose.pts[side["shoulder"]], pose.pts[side["elbow"]], pose.pts[side["wrist"]])

    def collect(self, pose):
        s = self._side
        sh, hip, ank = pose.pts[s["shoulder"]], pose.pts[s["hip"]], pose.pts[s["ankle"]]
        deviation = 180 - joint_angle(sh, hip, ank)  # 0 = straight shoulder-hip-ankle line
        # Where the hip sits relative to the shoulder->ankle line: below (larger y) = sagging
        t = np.dot(hip - sh, ank - sh) / (np.dot(ank - sh, ank - sh) + 1e-9)
        sagging = hip[1] > (sh + t * (ank - sh))[1]
        self.record("line_dev", deviation if sagging else -deviation)

    def evaluate(self):
        devs = np.array(self.metrics.get("line_dev", [0.0]))
        subs = {
            "depth": ramp(self.min_angle, 90, 130),               # 100% at <=90 deg elbow
            "body_line": ramp(np.percentile(np.abs(devs), 90), 10, 30),
            "lockout": ramp(self.lockout_angle, 165, 140),        # arms straight at the top
        }
        cues = []
        if self.min_angle > 95:
            cues.append("Lower your chest further")
        if subs["body_line"] < 70:
            cues.append("Hips sagging - brace your core" if devs.mean() > 0 else "Hips too high - flatten your body")
        if subs["lockout"] < 70:
            cues.append("Fully extend your arms at the top")
        return subs, cues


EXERCISES = {"squat": Squat, "pushup": PushUp}
