# Findings: CV Exercise Tracking (reps + form score)

**Audience:** Engineering & Product · **Spike date:** 2026-09-12 · **Prototype:** [prototype/](prototype/)

## TL;DR
- **Recommendation:** use an **on-device pose estimation model (Google MediaPipe BlazePose, 33 3D landmarks)**. Put **rule-based geometry** on top of it: joint angles → a rep state machine → a weighted form score.
- **No training data is needed** for the MVP. Every exercise is about 50 lines of rules (thresholds + scoring). Squats and push-ups are built and working in the prototype.
- Form feedback can be explained: every point lost maps to a cue like "go deeper" or "hips sagging". Product gets per-rep scores and text cues.
- **Runs on-device:** no video leaves the phone, costs nothing per user, and works offline. The desktop CPU runs about 25 ms per frame (~40 FPS inference). On a mid-range phone (Galaxy S24 FE, GPU delegate) the first measurement was **~47 ms inference and 10–13 FPS end-to-end**. That's slower than the desktop but enough for counting reps (§3d).
- **React Native (Expo):** built as a **small custom Expo native module** (Kotlin: CameraX + MediaPipe Tasks) instead of the community vision-camera plugins, which lag behind our Expo SDK 57 / React Native 0.86 stack (§6). It **needs an Expo dev build (not Expo Go)**. The rep and score logic is ported 1:1 to TypeScript and verified identical to the Python.
- **Main risks:** (1) camera placement. Users must be side-on, with their full body in frame. (2) We own the native camera/pose code, and iOS still needs its Swift counterpart (~1–2 days).
- **Building the next app?** Start at **§10**. It covers what to reuse as-is, the native design decisions that mattered, build gotchas, and the size/performance budget.

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

What the push-up session tells us about tuning. It's only one session, so these are hypotheses, not conclusions:

| Signal | Observation | Likely meaning → action |
|---|---|---|
| Depth | Scored 98.5–100 on all 6 reps | The 90°/130° elbow ramp may be too lenient. Check against deliberately shallow reps. |
| Lockout | Scored ~55 on reps 1–2, even though those reps were otherwise clean | 165° may be too strict for 2D angles after EMA smoothing, since arms rarely read fully straight. **First tuning candidate.** |
| Tempo | "Slow down" fired on reps 4–6 (0.47–0.8 s) | This matched genuinely fast reps. The 0.8 s floor looks right. |
| Detection | Pose found in only 69% of frames | A side-on plank at floor height often drops out of detection. This reinforces the need for placement onboarding and a "body visible" indicator (§7). |

### 3d. On-device Android (Expo app, custom module)
**Device:** Samsung Galaxy S24 FE (SM-S721U), Android 14 (API 34), Exynos 2400e. **Build:** standalone release APK (arm64). `pose_landmarker_full`, GPU delegate with CPU fallback. Camera frames are about 640×480. Timings come from the app's `SESSION_SUMMARY` log.

**First measurements (2026-09-12, full model, GPU):** about **47 ms** inference, about **8 ms** frame preprocessing (ImageProxy → rotated Bitmap), and **10–13 FPS** end-to-end. The rep-accuracy trials below are still to do.

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
| Logic | [mobile/src/logic/](mobile/src/logic/) is a 1:1 TypeScript port of `pose.py` + `exercises.py`, with the same thresholds, weights and cues. It runs on the JS thread for each pose event. `npm run test:logic` replays the §3a scenarios. Fed the same frames, the TypeScript and Python engines give **identical** reps, scores, sub-scores and cues. The app currently uses **2D angles only**. `worldLandmarks` cross the bridge but nothing uses them yet, and there's no 3D toggle in the UI. |
| UI | Exercise picker, camera screen with a `react-native-svg` skeleton overlay (cover-fit + front-camera mirroring), a HUD (reps, partial reps, angle, last/average %, cues, status, FPS, ms), and a per-rep summary table at the end. `expo-keep-awake` keeps the screen on during a set. |
| Build | A local dev build over USB (`npx expo run:android`), or a **standalone release APK** for untethered testing (§10.3). Expo Go can't load it. EAS isn't needed. `mobile/android/` is gitignored and regenerated by prebuild (CNG), so all native config lives in the module and `app.json`. |

**Why not the community plugins from the original plan:**
- `react-native-vision-camera` has moved to **v5, a Nitro Modules rewrite**. The MediaPipe frame-processor plugins were written for **v4 + `react-native-worklets-core`**.
- The strongest candidate, `react-native-mediapipe-posedetection`, targets vision-camera 4 on **React Native 0.81**. We're on **Expo SDK 57 / React Native 0.86**, so we'd be depending on three community packages that are behind our version, with no guarantee they build.
- The custom module is about 320 lines of Kotlin with two first-party Google dependencies whose versions we pin. Only 33 points per frame cross into JavaScript, so we don't need frame-processor worklets.
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
| We own the native camera/pose code | Upgrades (CameraX, MediaPipe, Expo) are on us, and **iOS isn't started** | The module is small, with two pinned first-party deps. iOS is a Swift view behind the same props/events (~1–2 days). |
| Thresholds tuned on few bodies | Scores feel wrong for some users (mobility, body proportions) | Collect feedback, make thresholds remote-configurable |
| Loose clothing / low light / cluttered background | Landmark jitter | EMA smoothing, visibility threshold, UX guidance |
| Battery / thermals | Long sessions drain the phone | `lite` model, 15 FPS cap |

**Open questions for product:** which exercises come after squats and push-ups? Is a single overall % enough, or do users want the sub-scores? Should partial reps be shown? Will we log (anonymised) landmark data to improve scoring later?

## 10. Lessons for the next app

### 10.1 Reuse as-is
| Piece | Path | Notes |
|---|---|---|
| Native camera + pose view | [mobile/modules/pose-camera/](mobile/modules/pose-camera/) | About 320 lines of Kotlin plus TS types. Drop it into any Expo SDK 57 app as a local module. The API is `<PoseCameraView model cameraFacing onPose onError />`. |
| Rep + score engine | [mobile/src/logic/](mobile/src/logic/) | `pose.ts` (geometry) and `exercises.ts` (state machine + scoring) have no dependencies. A new exercise is a subclass with `primaryAngle` / `collect` / `evaluate`. |
| Logic tests | `mobile/src/logic/selftest.ts` | `npm run test:logic` replays synthetic poses. `-- --dump` prints frames for cross-checking against Python. |
| Skeleton overlay | `SkeletonOverlay` in [mobile/App.tsx](mobile/App.tsx) | Cover-fit scaling + front-camera mirroring to draw landmarks over `PreviewView`. |
| Session logging | `SESSION_SUMMARY {json}` in App.tsx | One grep-able log line per session with reps, sub-scores, FPS, ms and delegate. Read it with `adb logcat -s ReactNativeJS PoseCamera`. |
| Desktop prototype | [prototype/](prototype/) | Fastest way to iterate on thresholds. `--video file.mp4` gives repeatable runs, and sessions save to JSON. |

### 10.2 Native design decisions that mattered
| Decision | Why |
|---|---|
| `shouldUseAndroidLayout = true` on the `ExpoView` | React Native doesn't lay out native children, so `PreviewView` stays blank without a real measure/layout pass. |
| Same 4:3 `AspectRatioStrategy` on Preview and ImageAnalysis | Landmarks and preview share one aspect ratio, so JS can map points with simple "object-fit: cover" math. |
| `STRATEGY_KEEP_ONLY_LATEST` + **one frame in flight** (1 s stuck-frame timeout) | The model never builds a queue, latency stays flat, `inferenceMs` is pure model time, and the event rate is the real pose FPS. |
| Strictly increasing timestamps (`max(now, last + 1)`) | MediaPipe LIVE_STREAM mode rejects equal or backwards timestamps. |
| Model loaded from module `assets/` into a **direct** `ByteBuffer` | `setModelAssetBuffer` needs a direct buffer. Call `rewind()` before retrying with another delegate. |
| GPU → CPU fallback **at init and at runtime** | GPU init can fail on some devices, and GPU errors can also surface later via the error listener. Both paths rebuild the landmarker on CPU. |
| Landmarks sent in rotated, **un-mirrored** image coords + a `mirrored` flag | Scoring math stays camera-independent. Only the overlay flips x. |
| Events go out via `post {}` with a `released` guard | Results arrive on the analysis thread and must be dispatched on the main thread. The guard prevents events after unmount. |
| Only 33 points per frame cross into JS | No frame-processor worklets needed. JS runs the engine on each `onPose` event. |

### 10.3 Build & tooling gotchas
- **Read the versioned Expo docs first** (`mobile/AGENTS.md`: https://docs.expo.dev/versions/v57.0.0/). Expo APIs changed, and older examples don't apply.
- **JDK:** run Gradle with `JAVA_HOME=/opt/android-studio/jbr` (Android Studio's JDK 21). The system `java-21-openjdk` is a JRE with no `javac`, and Gradle fails with *"does not provide JAVA_COMPILER"*.
- On the first build, Gradle auto-installs **NDK 27.1, platform 36 and build-tools 35**. Expect a slow first build.
- `npx expo run:android --device <x>` takes the device **name**, not the adb serial. With one phone connected, omit the flag.
- **Untethered testing needs a release APK.** A dev build needs Metro:
  ```
  JAVA_HOME=/opt/android-studio/jbr android/gradlew -p android app:assembleRelease -PreactNativeArchitectures=arm64-v8a   # ~5 min
  adb install -r android/app/build/outputs/apk/release/app-release.apk
  ```
  It's signed with the debug key, so it installs over the debug build without uninstalling.
- Expo Go can't load custom native modules, so you always need a dev or release build.
- Pinned versions that built together: Expo SDK 57, React Native 0.86.3, React 19.2, CameraX 1.6.2, MediaPipe `tasks-vision` 1.0.0, `react-native-svg` 15.15. Desktop: `mediapipe` 1.0.1, OpenCV 5.0.

### 10.4 Size & performance budget
| Where | Model | Inference | End-to-end | Other |
|---|---|---|---|---|
| Desktop CPU, empty frame | lite / full | ~26 / ~24 ms | – | Worst case (detector runs every frame) |
| Desktop webcam, live push-ups | full | 33.5 ms | 13.6 FPS | Pose found in 69% of frames |
| Galaxy S24 FE, GPU | full | ~47 ms | 10–13 FPS | ~8 ms preprocessing per frame |

- The **release APK (arm64 only) is ~55.7 MB**. It bundles both models: `full` 9.4 MB and `lite` 5.8 MB. To shrink it, ship one model, or download on first run the way the prototype does (`MODEL_URL` in `prototype/pose.py`, from Google's storage bucket).
- 10–15 FPS is enough for rep counting and form scoring. At 10 FPS, a 0.8 s rep (the push-up "Slow down" floor) still gives about 8 angle samples.

### 10.5 How we ported & verified the logic
- Keep all scoring as **pure functions over landmarks**, with no camera or UI imports. That's what made a 1:1 Python → TypeScript port possible.
- Verify parity by replaying the **same frames** through both engines and diffing the reps, scores, sub-scores and cues (`selftest.ts --dump` → `exercises.py`).
- Parity traps we hit:
  - Python's `max()` keeps the first of equal keys, so the **left side wins ties** in `bestSide`. TS must match.
  - `numpy.percentile` uses **linear interpolation** and was reimplemented by hand.

### 10.6 Known optimizations not yet done (unmeasured)
- Skip the Bitmap rotation in Kotlin (part of the ~8 ms preprocessing) by passing the rotation to MediaPipe via `ImageProcessingOptions`.
- Throttle or memoise React state updates. Right now every `onPose` event re-renders the whole camera screen (HUD + SVG overlay).
- Cap analysis at ~15 FPS for battery and thermals on long sessions (§4).
- Wire up the `3d` angle mode in the app for users who face the camera (it already exists in Python and in `poseFromLandmarks`).

## Next steps
1. Run the on-device trials (§3d) and the remaining webcam trials (§3c): ~10 good + ~5 bad reps per exercise.
2. Tune thresholds from that data, starting with push-up lockout and depth (§3c).
3. Design the camera-placement onboarding with a live "body visible" indicator.
4. Build the iOS Swift counterpart of `PoseCameraView`.
5. Product decision: do we log anonymised landmarks (with consent) to train a learned scorer or exercise auto-detection later (§8)?
