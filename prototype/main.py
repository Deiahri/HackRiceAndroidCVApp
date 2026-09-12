"""Live rep counter + form scorer prototype.

    python main.py --exercise squat
    python main.py --exercise pushup --model lite
    python main.py --exercise squat --angles 3d     # view-independent angles from 3D world landmarks

Keys: q / Esc = quit (prints + saves summary), r = reset counter.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2

from exercises import EXERCISES, Exercise
from pose import SKELETON, VIS_THRESHOLD, Pose, PoseEstimator

SESSIONS_DIR = Path(__file__).parent / "sessions"
WINDOW = "CV-Exercise prototype"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exercise", required=True, choices=EXERCISES)
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--video", help="run on a video file instead of the webcam")
    p.add_argument("--model", default="full", choices=["lite", "full", "heavy"], help="BlazePose variant")
    p.add_argument("--angles", default="2d", choices=["2d", "3d"],
                   help="2d = image-plane angles (needs side view); 3d = world landmarks")
    p.add_argument("--no-mirror", action="store_true", help="don't mirror the webcam image")
    return p.parse_args()


def put(frame, text, org, scale=0.7, color=(255, 255, 255)):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def score_color(score: float):
    return (80, 220, 80) if score >= 80 else (0, 220, 255) if score >= 60 else (60, 60, 255)


def draw_skeleton(frame, pose: Pose) -> None:
    pt = lambda i: tuple(int(v) for v in pose.px[i])
    for a, b in SKELETON:
        if pose.visible(a, b):
            cv2.line(frame, pt(a), pt(b), (255, 255, 255), 3)
    for i in {i for pair in SKELETON for i in pair}:
        cv2.circle(frame, pt(i), 5, (0, 220, 0) if pose.vis[i] >= VIS_THRESHOLD else (0, 0, 220), -1)


def draw_hud(frame, ex: Exercise, fps: float, model: str) -> None:
    h = frame.shape[0]
    put(frame, f"{ex.name.upper()}  reps: {len(ex.reps)}  (partial: {ex.partial_reps})", (15, 40), 1.0)
    angle = f"{ex.angle:.0f}" if ex.angle is not None else "-"
    put(frame, f"angle: {angle}  state: {ex.state}", (15, 75))
    if ex.reps:
        last = ex.reps[-1]
        put(frame, f"last rep: {last.score:.0f}%", (15, 110), 0.8, score_color(last.score))
        put(frame, f"avg: {ex.avg_score:.0f}%", (230, 110), 0.8, score_color(ex.avg_score))
        put(frame, " | ".join(last.cues), (15, 145), 0.7, (0, 220, 255))
    if ex.notice:
        put(frame, ex.notice, (15, 180), 0.7, (0, 140, 255))
    if ex.status:
        put(frame, ex.status, (15, h - 50), 0.9, (60, 60, 255))
    put(frame, f"{fps:.0f} FPS  model={model}", (15, h - 15), 0.6)


def print_summary(ex: Exercise, args, stats: dict) -> None:
    print(f"\n=== Session summary: {ex.name} (model={args.model}, angles={args.angles}) ===")
    print(f"Reps counted: {len(ex.reps)}   partial (not counted): {ex.partial_reps}")
    if ex.reps:
        keys = list(ex.weights)
        print(f"{'rep':>3} {'score':>6} " + " ".join(f"{k:>9}" for k in keys) + f" {'secs':>5}  cues")
        for r in ex.reps:
            subs = " ".join(f"{r.subscores[k]:>9.1f}" if k in r.subscores else f"{'-':>9}" for k in keys)
            print(f"{r.number:>3} {r.score:>6.1f} {subs} {r.duration_s:>5.2f}  {', '.join(r.cues)}")
        print(f"Average form score: {ex.avg_score:.1f}%")
    print(f"Avg FPS: {stats['avg_fps']:.1f}   avg inference: {stats['avg_inference_ms']:.1f} ms   "
          f"pose detected in {stats['pose_detected_pct']:.0f}% of frames")


def save_session(ex: Exercise, args, stats: dict) -> Path:
    SESSIONS_DIR.mkdir(exist_ok=True)
    path = SESSIONS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{ex.name}_{args.model}_{args.angles}.json"
    path.write_text(json.dumps({
        "exercise": ex.name, "model": args.model, "angles": args.angles,
        "source": args.video or f"camera:{args.camera}",
        "reps_counted": len(ex.reps), "partial_reps": ex.partial_reps, "avg_score": round(ex.avg_score, 1),
        "reps": [asdict(r) for r in ex.reps], "stats": stats,
    }, indent=2))
    return path


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video or f'camera {args.camera}'}")
    mirror = not args.video and not args.no_mirror
    estimator = PoseEstimator(args.model, use_3d=args.angles == "3d")
    ex = EXERCISES[args.exercise]()

    frames = detected = 0
    infer_total = fps = 0.0
    t_begin = last = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if args.video else now - t_begin

            t0 = time.perf_counter()
            pose = estimator.detect(frame, int(t * 1000))
            infer_total += time.perf_counter() - t0
            frames += 1
            detected += pose is not None

            rep = ex.update(pose, t)
            if rep:
                print(f"Rep {rep.number}: {rep.score:.0f}%  {', '.join(rep.cues)}")

            dt, last = now - last, now
            if dt > 0:
                fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
            if pose:
                draw_skeleton(frame, pose)
            draw_hud(frame, ex, fps, args.model)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("r"):
                ex = EXERCISES[args.exercise]()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        estimator.close()

    elapsed = time.perf_counter() - t_begin
    stats = {
        "frames": frames,
        "seconds": round(elapsed, 1),
        "avg_fps": round(frames / elapsed, 1) if elapsed else 0.0,
        "avg_inference_ms": round(1000 * infer_total / max(frames, 1), 1),
        "pose_detected_pct": round(100 * detected / max(frames, 1), 1),
    }
    print_summary(ex, args, stats)
    print(f"Saved: {save_session(ex, args, stats)}")


if __name__ == "__main__":
    main()
