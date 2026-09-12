# CV-Exercise prototype

This prototype counts reps and scores form from a webcam, using MediaPipe BlazePose plus rule-based geometry. See [../Findings.md](../Findings.md) for the write-up.

## Setup
```bash
uv venv .venv --python 3.12 && uv pip install -p .venv -r requirements.txt
# or: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```
The pose model (`models/pose_landmarker_<variant>.task`) downloads automatically on the first run.

## Run
```bash
.venv/bin/python main.py --exercise squat
.venv/bin/python main.py --exercise pushup
```
| Flag | Default | Notes |
|---|---|---|
| `--model lite\|full\|heavy` | full | BlazePose variant: speed vs. accuracy |
| `--angles 2d\|3d` | 2d | `3d` uses world landmarks, so the camera angle matters less |
| `--camera N` / `--video file.mp4` | camera 0 | video file for repeatable tests |

Keys: `q`/`Esc` quits and prints a summary, which is also saved to `sessions/*.json`. `r` resets the counter.

## Camera placement
- **Push-ups:** side-on to the camera, with your whole body (head to ankles) in frame, at floor or low table height.
- **Squats:** side-on or at 45°, with your full body in frame. Facing the camera head-on hides knee bend in 2D mode. Use `--angles 3d` if you face the camera.
- Good lighting helps, and so does a plain background with nothing blocking your legs.

## Files
- `pose.py`: PoseLandmarker wrapper plus geometry helpers (joint angle, angle from vertical, EMA smoothing).
- `exercises.py`: rep state machine and per-exercise form scoring (`Squat`, `PushUp`).
- `main.py`: webcam loop, overlay, and session summary.
