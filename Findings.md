# Findings: CV Exercise Tracking (reps + form score)

**Audience:** Engineering & Product · **Spike date:** 2026-09-12 · **Prototype:** [prototype/](prototype/)

## TL;DR
- **Recommendation:** use an **on-device pose estimation model (Google MediaPipe BlazePose, 33 3D landmarks)**. Put **rule-based geometry** on top of it: joint angles → a rep state machine → a weighted form score.
- **No training data is needed** for the MVP. Every exercise is about 50 lines of rules (thresholds + scoring). Squats and push-ups are built and working in the prototype.
- Form feedback can be explained: every point lost maps to a cue like "go deeper" or "hips sagging". Product gets per-rep scores and text cues.
- **Runs on-device:** no video leaves the phone, costs nothing per user, and works offline. The desktop CPU runs about 25 ms per frame (~40 FPS inference). Phones with GPU delegates are in the same range.
- **React Native (Expo):** built as a **small custom Expo native module** (Kotlin: CameraX + MediaPipe Tasks) instead of the community vision-camera plugins, which lag behind our Expo SDK 57 / React Native 0.86 stack (§6). It **needs an Expo dev build (not Expo Go)**. The rep and score logic is ported 1:1 to TypeScript and verified identical to the Python.
- **Main risks:** (1) camera placement. Users must be side-on, with their full body in frame. (2) We own the native camera/pose code, and iOS still needs its Swift counterpart (~1–2 days).

---

## 1. Model type

| Layer | What it is | Trained? |
|---|---|---|
| **Pose estimation** | **MediaPipe Pose Landmarker (BlazePose)**. Two stages: a person detector, then a CNN that regresses **33 body landmarks** (x, y, visibility, plus 3D "world" coords in metres). It tracks the previous frame's region, so the detector rarely re-runs. Variants: `lite` / `full` / `heavy`. | Pretrained by Google, used as-is |
| **Rep counting** | A finite state machine driven by one joint angle: the knee for squats, the elbow for push-ups. Hysteresis thresholds reject jitter. | Rules |
| **Form scoring** | Per-rep sub-scores (0–100) from angle metrics, weighted into one %. Each failed check emits a coaching cue. | Rules |
| **Exercise type** | The user selects it (MVP). See §8 for auto-detection. | – |

**Why pose + rules, not an end-to-end "video → score" model?**
- End-to-end action models (e.g. video transformers, temporal CNNs on raw frames) need labelled video datasets of good and bad reps for *every* exercise. They're heavy on mobile, and they give a score with no *reason*.
- Pose keypoints compress each frame into 33 points. That makes reps and form a geometry problem, which is cheap, deterministic, testable and explainable. Most shipping fitness apps use this pattern.

## 2. How it works

```
camera frame ─► BlazePose ─► 33 landmarks ─► pick most-visible body side ─► joint angles (EMA-smoothed)
                                                                              │
             ┌────────────────────────────────────────────────────────────────┘
             ▼
  waiting ─(angle > UP)─► top ─(angle < UP-10°)─► descent ─(angle < DOWN)─► bottom ─(angle > UP)─► top  ✔ rep counted + scored
                                                     └────────(angle > UP, never hit DOWN)──────► top  ✖ "partial rep"
```

| | Squat | Push-up |
|---|---|---|
| Driving angle | knee (hip–knee–ankle) | elbow (shoulder–elbow–wrist) |
| Top / bottom thresholds | > 160° / < 110° | > 150° / < 100° |
| Guard | – | body must be ≥ 45° from vertical (in plank) so arm bends while standing don't count |
| Sub-scores (weight) | **depth** 50%: 100% at knee ≤ 90°, 50% at 110°<br>**torso lean** 30%: 100% ≤ 45°, 0% ≥ 75°<br>**L/R symmetry** 20%: 100% ≤ 10° diff, 0% ≥ 35° | **depth** 40%: 100% at elbow ≤ 90°<br>**body line** 40%: shoulder–hip–ankle deviation 100% ≤ 10°, 0% ≥ 30°, sag vs. pike detected<br>**lockout** 20%: arm straightness at top, 100% ≥ 165° |
| Cues | "Go deeper", "Keep your chest up", "Uneven legs", "Slow down" | "Lower your chest further", "Hips sagging", "Hips too high", "Fully extend your arms", "Slow down" |

- A sub-score that can't be measured (e.g. a leg is occluded) is dropped, and the remaining weights are re-normalised.
- The session score is the mean of the per-rep scores.
- Two angle modes: `2d` (image plane; needs a side view) and `3d` (BlazePose world landmarks; less sensitive to camera angle, but noisier depth).

These thresholds are **starting values** and should be tuned against real users. They are constants, so product/fitness experts can own them.

## 3. Prototype results

### 3a. Logic tests (synthetic poses)
| Scenario | Expected | Result |
|---|---|---|
| Squat: 3 good reps | 3 × 100% | ✅ 3 × 100% |
| Squat: shallow (135°) | not counted, flagged partial | ✅ partial = 1 |
| Squat: 105° depth | counted, depth penalised | ✅ 79.7%, "Go deeper" |
| Squat: 65° torso lean | counted, torso penalised | ✅ 80.0%, "Keep your chest up" |
| Push-up: 3 good reps | 3 × 100% | ✅ 3 × 100% |
| Push-up: hips sagging | body-line penalised | ✅ 60.0%, "Hips sagging" |
| Push-up: shallow (115°) | not counted, flagged partial | ✅ partial = 1 |

### 3b. Performance (desktop CPU, MediaPipe 1.0.1, 640×480)
| Model | Inference / frame |
|---|---|
| lite | ~26 ms |
| full | ~24 ms |

(These were measured on an empty frame, which runs the detector every frame and so is the worst case. Tracking a person is usually faster.)

### 3c. Live webcam trials (desktop, Python prototype)
The team ran the prototype live on a laptop webcam. Only one session was saved to `prototype/sessions/`: `20260912-024658_pushup_full_2d.json`.

| Exercise | Setup | Actual reps | Counted | Avg score | Notes |
|---|---|---|---|---|---|
| Push-up | mixed good + deliberately bad reps, side view, `full`, `2d` | _to confirm_ | 6 (0 partial) | 80.6% | 13.6 FPS end-to-end, 33.5 ms inference, pose found in 69% of frames. Per-rep scores 90 / 86 / 94 / 85 / 56 / 72. The cues matched the faults: "Fully extend your arms" (lockout ~55%), "Hips sagging" (body line 0%), "Hips too high", and "Slow down" on reps under 0.8 s. |
| Squat | good form / shallow / facing camera `--angles 3d` | – | – | – | Not saved |

### 3d. On-device Android (Expo app, custom module)
**Device:** Samsung Galaxy S24 FE (SM-S721U), Android 14 (API 34), Exynos 2400e. **Build:** standalone release APK (arm64). `pose_landmarker_full`, GPU delegate with CPU fallback. Camera frames are about 640×480. Timings come from the app's `SESSION_SUMMARY` log.

| Exercise | Setup | Actual reps | Counted | Partial | Avg score | FPS | Inference ms | Delegate | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Squat | good form, side view | _pending_ | | | | | | | |
| Squat | shallow / leaning | _pending_ | | | | | | | |
| Push-up | good form, side view | _pending_ | | | | | | | |
| Push-up | hips sagging / shallow | _pending_ | | | | | | | |

## 4. Tradeoffs

| Decision | Upside | Downside / mitigation |
|---|---|---|
| **Pose + rules** (vs. learned scoring) | No data needed, explainable cues, deterministic, easy to unit-test | Every exercise needs hand-tuned rules. Rules miss subtle faults (e.g. knee valgus). **Mitigation:** log landmarks (not video) to later train a learned scorer. |
| **On-device** (vs. cloud inference) | Privacy (no video uploaded), zero marginal cost, works offline, low latency for live feedback | Battery and heat on long sessions. Accuracy is limited by phone compute. **Mitigation:** `lite` model on low-end devices, and lower the frame rate to 15 FPS (enough for reps). |
| **BlazePose lite / full / heavy** | lite is fastest, heavy is most accurate | Pick `full` as the default and `lite` as the fallback. `heavy` is mostly not worth it on mobile. |
| **2D vs. 3D angles** | 2D is stable and accurate from a side view | 2D breaks when the user faces the camera (knee bend is invisible). 3D handles any view, but its depth estimate is noisy. **Mitigation:** onboarding UI to guide camera placement. |
| **Single-camera, single-person** | Simple UX | Occlusion (the far arm or leg is hidden in a side view) is handled by using the most-visible side. Only one person is tracked. |
| **User-selected exercise** | Removes a whole ML problem from the MVP | Extra tap for the user. Auto-detection is a later add-on (§8). |

## 5. Alternatives considered

### Pose models
| Option | Keypoints | 3D | Mobile / RN fit | License | Verdict |
|---|---|---|---|---|---|
| **MediaPipe BlazePose** (recommended) | 33 (incl. feet, hands) | ✅ world coords | Native iOS/Android/Web SDKs. Several RN vision-camera plugins | Apache-2.0 | **Best balance.** Foot and heel points help squats. |
| Google **ML Kit Pose** | 33 (BlazePose-based) | z-coord | Easy native SDK. RN vision-camera plugins exist | Free (proprietary SDK) | **Strong fallback.** Same model family, simpler API, less control over the model version. |
| **MoveNet** Lightning / Thunder | 17 (COCO) | ❌ | TFLite / TF.js, very fast | Apache-2.0 | Good for speed. No feet and no 3D, which hurts squat and push-up checks. |
| **Apple Vision** body pose | 19 (2D), 3D on iOS 17+ | iOS 17+ | iOS only | Apple | Not cross-platform. |
| **YOLO-pose** (Ultralytics v8/11) | 17 | ❌ | Exportable to TFLite/CoreML, multi-person | **AGPL-3.0** (commercial licence needed) | Licence risk. Multi-person isn't needed. |
| **RTMPose** (MMPose) | 17 / 133 | ❌ (2D) | ONNX/ncnn, needs custom native work | Apache-2.0 | More accurate, but much more integration effort. |
| **OpenPose** | 25 | ❌ | Too heavy for mobile | Non-commercial | ❌ |
| **Cloud inference** (own GPU server / hosted vision API) | varies | varies | Upload video stream | – | ❌ Latency, cost per minute, privacy. Only useful for offline post-workout analysis. |

### Rep counting / form scoring approaches
| Approach | Data needed | Pros | Cons | When |
|---|---|---|---|---|
| **Rules on joint angles** (chosen) | none | Explainable, fast to ship | Manual tuning per exercise | **MVP** |
| kNN / classifier on pose embeddings (MediaPipe "pose classification" recipe) | ~tens of labelled poses per state | Robust up/down states, little data | Still needs rules for scoring | If angle thresholds prove brittle |
| Temporal model (LSTM / 1D-CNN / transformer) on landmark sequences | 100s–1000s of labelled reps with quality scores | Learns subtle faults, one model for many exercises | Data collection + labelling, harder to explain | v2, after logging real sessions |
| Periodicity / signal-based counting (autocorrelation on a joint trajectory) | none | Exercise-agnostic counting | No form score, struggles with pauses | Possible "any exercise" counter |

## 6. React Native (Expo) integration path

**Chosen and built ([mobile/](mobile/)):** a small **local Expo Module** in Kotlin ([mobile/modules/pose-camera/](mobile/modules/pose-camera/)). It wraps Google's official MediaPipe Tasks SDK directly, with no third-party camera or pose plugin. It's Android-only for now.

| Piece | What it is |
|---|---|
| Camera | **CameraX 1.6.2**. `PreviewView` shows the preview. `ImageAnalysis` delivers RGBA_8888 frames at about 640×480 and keeps only the latest frame. Front camera by default, switchable. |
| Pose | **MediaPipe `tasks-vision` 1.0.0** `PoseLandmarker` in LIVE_STREAM mode. **GPU delegate, falling back to CPU.** The model is bundled in the APK: `full` by default, `lite` as an option. One frame is in flight at a time, so the model never builds a queue. |
| Bridge | One native view: `<PoseCameraView model cameraFacing onPose onError />`. Each frame, `onPose` sends 33 landmarks (normalized x/y/z + visibility), the world landmarks, the image size, inference ms and a mirrored flag. |
| Logic | [mobile/src/logic/](mobile/src/logic/) is a 1:1 TypeScript port of `pose.py` + `exercises.py`, with the same thresholds, weights and cues. It runs on the JS thread for each pose event. `npm run test:logic` replays the §3a scenarios. Fed the same frames, the TypeScript and Python engines give **identical** reps, scores, sub-scores and cues. |
| UI | Exercise picker, camera screen with a `react-native-svg` skeleton overlay (cover-fit + front-camera mirroring), a HUD (reps, partial reps, angle, last/average %, cues, status, FPS, ms), and a per-rep summary table at the end. `expo-keep-awake` keeps the screen on during a set. |
| Build | A local dev build over USB: `npx expo run:android`. Expo Go can't load it. EAS isn't needed. |

**Why not the community plugins from the original plan:**
- `react-native-vision-camera` has moved to **v5, a Nitro Modules rewrite**. The MediaPipe frame-processor plugins were written for **v4 + `react-native-worklets-core`**.
- The strongest candidate, `react-native-mediapipe-posedetection`, targets vision-camera 4 on **React Native 0.81**. We're on **Expo SDK 57 / React Native 0.86**, so we'd be depending on three community packages that are behind our version, with no guarantee they build.
- The custom module is about 250 lines of Kotlin with two first-party Google dependencies whose versions we pin. Only 33 points per frame cross into JavaScript, so we don't need frame-processor worklets.
- **The cost:** we own the native code. **iOS** needs its own Swift view (AVFoundation + the `MediaPipeTasksVision` pod) behind the same props and events. Our estimate is **1–2 dev days**, because the JavaScript logic and UI are shared.

<details><summary>Community plugins we evaluated (for reference)</summary>

- [`react-native-mediapipe`](https://cdiddy77.github.io/react-native-mediapipe/docs/api_pages/pose-landmark-detection/) (pose landmark detection API)
- [`react-native-mediapipe-posedetection`](https://github.com/EndLess728/react-native-mediapipe-posedetection) (BlazePose, GPU, world coords; New Architecture only; vision-camera 4)
- [`@gymbrosinc/react-native-mediapipe-pose`](https://www.npmjs.com/package/@gymbrosinc/react-native-mediapipe-pose)
- [`expo-pose-landmarks`](https://www.npmjs.com/package/expo-pose-landmarks)
- ML Kit route: [`react-native-vision-camera-v3-pose-detection`](https://github.com/gev2002/react-native-vision-camera-v3-pose-detection). [`react-native-vision-camera-mlkit`](https://github.com/pedrol2b/react-native-vision-camera-mlkit) was noted as not yet iOS-ready.
- Reference for the custom-module approach: [Expo pose detection demo](https://github.com/mantu-bit/Expo-React-native-pose-detection-demo).
</details>

## 7. Product / UX implications
- **Onboarding for camera placement is essential:** prop the phone up and stand side-on, with your full body in frame. Show a live "body visible ✓" indicator. The prototype already shows "Body not fully visible" / "Get into a plank".
- A **"partial rep, not counted"** state is useful feedback. Decide whether partial reps are shown to users.
- Scores are **relative, not clinical**. Frame them as "form score" with cues, not as medical or physio advice.
- Rep-by-rep cues can be spoken (TTS) so users don't need to look at the phone mid-set.

## 8. Future: auto-detecting the exercise
- Run a lightweight classifier over a rolling window of ~1–2 s of landmarks: kNN on normalised pose embeddings for a quick win, or a small 1D-CNN/GRU for robustness.
- Data needed: roughly 50–100 short clips per exercise from varied people and camera angles, plus "idle/other" clips.
- Rules can bootstrap the labels: sessions recorded in manual-selection mode become training data for free if we log landmarks, with consent.

## 9. Risks & open questions
| Risk | Impact | Mitigation |
|---|---|---|
| Camera angle / placement varies wildly | Miscounts, unfair scores | Onboarding + visibility gating + `3d` angles fallback |
| RN plugin maturity | Integration delays | Evaluate early. Plan B native Expo module |
| Thresholds tuned on few bodies | Scores feel wrong for some users (mobility, body proportions) | Collect feedback, make thresholds remote-configurable |
| Loose clothing / low light / cluttered background | Landmark jitter | EMA smoothing, visibility threshold, UX guidance |
| Battery / thermals | Long sessions drain the phone | `lite` model, 15 FPS cap |

**Open questions for product:** which exercises come after squats and push-ups? Is a single overall % enough, or do users want the sub-scores? Should partial reps be shown? Will we log (anonymised) landmark data to improve scoring later?

## Next steps
1. Finish the live webcam trials (§3c) and tune thresholds.
2. Run the 1-day RN plugin evaluation on real devices.
3. Port the rep/score engine to TypeScript, with recorded-landmark unit tests.
4. Design the camera-placement onboarding.
