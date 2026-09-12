import { useKeepAwake } from 'expo-keep-awake';
import { StatusBar } from 'expo-status-bar';
import { useCallback, useRef, useState } from 'react';
import {
  LayoutChangeEvent, PermissionsAndroid, Pressable, ScrollView, StatusBar as RNStatusBar, StyleSheet, Text, View,
} from 'react-native';
import Svg, { Circle, Line } from 'react-native-svg';

import { Landmark, PoseCameraView, PoseEventPayload } from './modules/pose-camera';
import { EXERCISES, ExerciseName, RepResult } from './src/logic/exercises';
import { SKELETON, VIS_THRESHOLD, poseFromLandmarks } from './src/logic/pose';

type Model = 'full' | 'lite';
type Facing = 'front' | 'back';
type Settings = { exercise: ExerciseName; model: Model; facing: Facing };

type SessionSummary = {
  exercise: ExerciseName;
  model: string;
  delegate: string;
  cameraFacing: Facing;
  repsCounted: number;
  partialReps: number;
  avgScore: number;
  subscoreKeys: string[];
  reps: RepResult[];
  stats: {
    frames: number;
    seconds: number;
    avgFps: number;
    avgInferenceMs: number;
    avgPreprocessMs: number;
    poseDetectedPct: number;
  };
};

const TOP_INSET = (RNStatusBar.currentHeight ?? 24) + 8;
const BOTTOM_INSET = 40;

const PLACEMENT: Record<ExerciseName, string> = {
  squat: 'Prop the phone up, stand side-on (or at 45°), whole body in frame.',
  pushup: 'Phone at floor height, side-on, head to ankles in frame.',
};

const scoreColor = (s: number) => (s >= 80 ? '#3ddc84' : s >= 60 ? '#ffc53d' : '#ff5c5c');
const round1 = (x: number) => Math.round(x * 10) / 10;

export default function App() {
  const [settings, setSettings] = useState<Settings>({ exercise: 'squat', model: 'full', facing: 'front' });
  const [screen, setScreen] = useState<'home' | 'camera' | 'summary'>('home');
  const [summary, setSummary] = useState<SessionSummary | null>(null);
  const [permissionError, setPermissionError] = useState('');

  const start = async (exercise: ExerciseName) => {
    const result = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.CAMERA, {
      title: 'Camera access',
      message: 'The camera is used to track your reps. Video never leaves the phone.',
      buttonPositive: 'OK',
    });
    if (result !== PermissionsAndroid.RESULTS.GRANTED) {
      setPermissionError('Camera permission is required to track reps.');
      return;
    }
    setPermissionError('');
    setSettings((s) => ({ ...s, exercise }));
    setScreen('camera');
  };

  if (screen === 'camera') {
    return (
      <CameraScreen
        settings={settings}
        onEnd={(s) => {
          setSummary(s);
          setScreen('summary');
        }}
      />
    );
  }
  if (screen === 'summary' && summary) {
    return <SummaryScreen summary={summary} onAgain={() => setScreen('camera')} onHome={() => setScreen('home')} />;
  }
  return (
    <HomeScreen
      settings={settings}
      onChange={(patch) => setSettings((s) => ({ ...s, ...patch }))}
      onStart={start}
      error={permissionError}
    />
  );
}

// --- Home ---------------------------------------------------------------------

function HomeScreen(props: {
  settings: Settings;
  onChange: (patch: Partial<Settings>) => void;
  onStart: (exercise: ExerciseName) => void;
  error: string;
}) {
  const { settings, onChange, onStart, error } = props;
  return (
    <View style={[styles.home, { paddingTop: TOP_INSET + 24 }]}>
      <StatusBar style="dark" />
      <Text style={styles.title}>CV Exercise</Text>
      <Text style={styles.subtitle}>Pick an exercise. Reps are counted and each rep gets a form score.</Text>

      {(['squat', 'pushup'] as ExerciseName[]).map((name) => (
        <Pressable key={name} style={({ pressed }) => [styles.card, pressed && styles.pressed]} onPress={() => onStart(name)}>
          <Text style={styles.cardTitle}>{name === 'squat' ? 'Squat' : 'Push-up'}</Text>
          <Text style={styles.cardHint}>{PLACEMENT[name]}</Text>
        </Pressable>
      ))}

      <Segmented
        label="Model"
        options={[['full', 'Full (accurate)'], ['lite', 'Lite (fast)']]}
        value={settings.model}
        onChange={(model) => onChange({ model: model as Model })}
      />
      <Segmented
        label="Camera"
        options={[['front', 'Front'], ['back', 'Back']]}
        value={settings.facing}
        onChange={(facing) => onChange({ facing: facing as Facing })}
      />
      {error ? <Text style={styles.error}>{error}</Text> : null}
    </View>
  );
}

function Segmented(props: { label: string; options: [string, string][]; value: string; onChange: (v: string) => void }) {
  return (
    <View style={styles.segmentRow}>
      <Text style={styles.segmentLabel}>{props.label}</Text>
      <View style={styles.segment}>
        {props.options.map(([value, text]) => (
          <Pressable
            key={value}
            onPress={() => props.onChange(value)}
            style={[styles.segmentItem, props.value === value && styles.segmentItemActive]}
          >
            <Text style={[styles.segmentText, props.value === value && styles.segmentTextActive]}>{text}</Text>
          </Pressable>
        ))}
      </View>
    </View>
  );
}

// --- Camera -------------------------------------------------------------------

type Frame = {
  landmarks: Landmark[];
  imageWidth: number;
  imageHeight: number;
  mirrored: boolean;
};

function CameraScreen({ settings, onEnd }: { settings: Settings; onEnd: (s: SessionSummary) => void }) {
  useKeepAwake();
  const [facing, setFacing] = useState<Facing>(settings.facing);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [frame, setFrame] = useState<Frame | null>(null);
  const [perf, setPerf] = useState({ fps: 0, inferenceMs: 0, delegate: '', model: settings.model as string });
  const [error, setError] = useState('');
  const [, setTick] = useState(0);

  const exRef = useRef(EXERCISES[settings.exercise]());
  const stats = useRef({ frames: 0, detected: 0, inferSum: 0, preSum: 0, fps: 0, last: 0, begin: 0 });
  const ex = exRef.current;

  const onPose = useCallback((e: { nativeEvent: PoseEventPayload }) => {
    const p = e.nativeEvent;
    const s = stats.current;
    const now = Date.now();
    if (s.begin === 0) s.begin = now;
    if (s.last) {
      const dt = (now - s.last) / 1000;
      if (dt > 0) s.fps = s.fps === 0 ? 1 / dt : 0.9 * s.fps + 0.1 / dt;
    }
    s.last = now;
    s.frames += 1;
    s.detected += p.landmarks.length ? 1 : 0;
    s.inferSum += p.inferenceMs;
    s.preSum += p.preprocessMs;

    const pose = poseFromLandmarks(p.landmarks, p.imageWidth, p.imageHeight);
    const rep = exRef.current.update(pose, p.timestampMs / 1000);
    if (rep) console.log(`Rep ${rep.number}: ${rep.score}%  ${rep.cues.join(', ')}`);

    setFrame({ landmarks: p.landmarks, imageWidth: p.imageWidth, imageHeight: p.imageHeight, mirrored: p.mirrored });
    setPerf({ fps: s.fps, inferenceMs: p.inferenceMs, delegate: p.delegate, model: p.model });
  }, []);

  const end = () => {
    const s = stats.current;
    const e = exRef.current;
    const seconds = s.begin ? (Date.now() - s.begin) / 1000 : 0;
    const summary: SessionSummary = {
      exercise: settings.exercise,
      model: perf.model,
      delegate: perf.delegate,
      cameraFacing: facing,
      repsCounted: e.reps.length,
      partialReps: e.partialReps,
      avgScore: round1(e.avgScore),
      subscoreKeys: Object.keys(e.weights),
      reps: e.reps,
      stats: {
        frames: s.frames,
        seconds: round1(seconds),
        avgFps: seconds ? round1(s.frames / seconds) : 0,
        avgInferenceMs: round1(s.inferSum / Math.max(s.frames, 1)),
        avgPreprocessMs: round1(s.preSum / Math.max(s.frames, 1)),
        poseDetectedPct: round1((100 * s.detected) / Math.max(s.frames, 1)),
      },
    };
    // Grep-able in `adb logcat -s ReactNativeJS` for the findings write-up.
    console.log(`SESSION_SUMMARY ${JSON.stringify(summary)}`);
    onEnd(summary);
  };

  const reset = () => {
    exRef.current = EXERCISES[settings.exercise]();
    stats.current = { frames: 0, detected: 0, inferSum: 0, preSum: 0, fps: 0, last: 0, begin: 0 };
    setTick((t) => t + 1);
  };

  const onLayout = (e: LayoutChangeEvent) => setSize(e.nativeEvent.layout);
  const last = ex.reps.at(-1);

  return (
    <View style={styles.camera} onLayout={onLayout}>
      <StatusBar style="light" />
      <PoseCameraView
        style={StyleSheet.absoluteFill}
        model={settings.model}
        cameraFacing={facing}
        onPose={onPose}
        onError={(e) => setError(e.nativeEvent.message)}
      />
      {frame && size.width > 0 ? <SkeletonOverlay frame={frame} width={size.width} height={size.height} /> : null}

      <View style={[styles.hud, { paddingTop: TOP_INSET }]} pointerEvents="none">
        <Text style={styles.hudTitle}>
          {ex.label.toUpperCase()}  <Text style={styles.hudBig}>{ex.reps.length}</Text> reps
          <Text style={styles.hudDim}>  (partial {ex.partialReps})</Text>
        </Text>
        <Text style={styles.hudText}>
          angle {ex.angle !== null ? `${Math.round(ex.angle)}°` : '–'}  ·  {ex.state}
        </Text>
        {last ? (
          <Text style={styles.hudText}>
            last <Text style={{ color: scoreColor(last.score), fontWeight: '700' }}>{Math.round(last.score)}%</Text>
            {'   '}avg <Text style={{ color: scoreColor(ex.avgScore), fontWeight: '700' }}>{Math.round(ex.avgScore)}%</Text>
          </Text>
        ) : null}
        {last ? <Text style={styles.cue}>{last.cues.join('  ·  ')}</Text> : null}
        {ex.notice ? <Text style={styles.notice}>{ex.notice}</Text> : null}
        {error ? <Text style={styles.error}>{error}</Text> : null}
      </View>

      <View style={[styles.bottom, { paddingBottom: BOTTOM_INSET }]}>
        {ex.status ? <Text style={styles.status}>{ex.status}</Text> : null}
        <Text style={styles.perf}>
          {perf.fps.toFixed(0)} FPS  ·  {perf.inferenceMs.toFixed(0)} ms  ·  {perf.model}
          {perf.delegate ? `/${perf.delegate}` : ''}
        </Text>
        <View style={styles.buttons}>
          <Pressable style={styles.button} onPress={() => setFacing((f) => (f === 'front' ? 'back' : 'front'))}>
            <Text style={styles.buttonText}>Flip</Text>
          </Pressable>
          <Pressable style={styles.button} onPress={reset}>
            <Text style={styles.buttonText}>Reset</Text>
          </Pressable>
          <Pressable style={[styles.button, styles.buttonPrimary]} onPress={end}>
            <Text style={[styles.buttonText, styles.buttonPrimaryText]}>End session</Text>
          </Pressable>
        </View>
      </View>
    </View>
  );
}

/** Landmarks are normalized to the camera image; the preview fills the view like CSS object-fit: cover. */
function SkeletonOverlay({ frame, width, height }: { frame: Frame; width: number; height: number }) {
  const { landmarks, imageWidth: w, imageHeight: h, mirrored } = frame;
  if (landmarks.length === 0 || !w || !h) return null;
  const scale = Math.max(width / w, height / h);
  const dx = (width - w * scale) / 2;
  const dy = (height - h * scale) / 2;
  const pt = (i: number) => {
    const x = landmarks[i].x * w * scale + dx;
    return { x: mirrored ? width - x : x, y: landmarks[i].y * h * scale + dy };
  };
  const visible = (i: number) => landmarks[i].visibility >= VIS_THRESHOLD;
  const joints = [...new Set(SKELETON.flat())];

  return (
    <Svg style={StyleSheet.absoluteFill} width={width} height={height} pointerEvents="none">
      {SKELETON.filter(([a, b]) => visible(a) && visible(b)).map(([a, b]) => {
        const p = pt(a), q = pt(b);
        return <Line key={`${a}-${b}`} x1={p.x} y1={p.y} x2={q.x} y2={q.y} stroke="white" strokeWidth={4} strokeLinecap="round" />;
      })}
      {joints.map((i) => {
        const p = pt(i);
        return <Circle key={i} cx={p.x} cy={p.y} r={6} fill={visible(i) ? '#3ddc84' : '#ff5c5c'} />;
      })}
    </Svg>
  );
}

// --- Summary ------------------------------------------------------------------

function SummaryScreen({ summary, onAgain, onHome }: { summary: SessionSummary; onAgain: () => void; onHome: () => void }) {
  const { stats, subscoreKeys: keys } = summary;
  const label = summary.exercise === 'squat' ? 'Squat' : 'Push-up';
  return (
    <ScrollView style={styles.summary} contentContainerStyle={{ paddingTop: TOP_INSET + 16, paddingBottom: BOTTOM_INSET + 16 }}>
      <StatusBar style="dark" />
      <Text style={styles.title}>{label} session</Text>
      <View style={styles.statRow}>
        <Stat label="Reps" value={String(summary.repsCounted)} />
        <Stat label="Partial" value={String(summary.partialReps)} />
        <Stat label="Avg form" value={summary.reps.length ? `${Math.round(summary.avgScore)}%` : '–'}
          color={summary.reps.length ? scoreColor(summary.avgScore) : undefined} />
      </View>

      <ScrollView horizontal style={styles.tableWrap}>
        <View>
          <View style={[styles.tr, styles.th]}>
            {['#', 'score', ...keys, 'secs', 'cues'].map((c, i) => (
              <Text key={c} style={[styles.td, styles.thText, i === keys.length + 3 && styles.tdWide]}>{c}</Text>
            ))}
          </View>
          {summary.reps.map((r) => (
            <View key={r.number} style={styles.tr}>
              <Text style={styles.td}>{r.number}</Text>
              <Text style={[styles.td, { color: scoreColor(r.score), fontWeight: '700' }]}>{r.score.toFixed(0)}%</Text>
              {keys.map((k) => (
                <Text key={k} style={styles.td}>{k in r.subscores ? r.subscores[k].toFixed(0) : '–'}</Text>
              ))}
              <Text style={styles.td}>{r.durationS.toFixed(1)}</Text>
              <Text style={[styles.td, styles.tdWide]}>{r.cues.join(', ')}</Text>
            </View>
          ))}
          {summary.reps.length === 0 ? <Text style={styles.empty}>No reps counted.</Text> : null}
        </View>
      </ScrollView>

      <Text style={styles.perfSummary}>
        {stats.avgFps} FPS avg · {stats.avgInferenceMs} ms inference · {stats.avgPreprocessMs} ms preprocess{'\n'}
        pose found in {stats.poseDetectedPct}% of {stats.frames} frames · {stats.seconds}s{'\n'}
        model {summary.model} / {summary.delegate} · {summary.cameraFacing} camera
      </Text>

      <View style={styles.buttonsLight}>
        <Pressable style={[styles.button, styles.buttonPrimary]} onPress={onAgain}>
          <Text style={[styles.buttonText, styles.buttonPrimaryText]}>Go again</Text>
        </Pressable>
        <Pressable style={[styles.button, styles.buttonOutline]} onPress={onHome}>
          <Text style={[styles.buttonText, { color: '#111' }]}>Change exercise</Text>
        </Pressable>
      </View>
    </ScrollView>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <View style={styles.stat}>
      <Text style={[styles.statValue, color ? { color } : null]}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

const shadow = { textShadowColor: 'rgba(0,0,0,0.85)', textShadowOffset: { width: 0, height: 1 }, textShadowRadius: 4 };

const styles = StyleSheet.create({
  home: { flex: 1, backgroundColor: '#f6f7f9', paddingHorizontal: 20 },
  title: { fontSize: 30, fontWeight: '800', color: '#111', paddingHorizontal: 4 },
  subtitle: { fontSize: 15, color: '#555', marginTop: 6, marginBottom: 20, paddingHorizontal: 4 },
  card: { backgroundColor: '#fff', borderRadius: 16, padding: 20, marginBottom: 14, elevation: 2 },
  pressed: { opacity: 0.7 },
  cardTitle: { fontSize: 22, fontWeight: '700', color: '#111' },
  cardHint: { fontSize: 14, color: '#666', marginTop: 6 },
  segmentRow: { flexDirection: 'row', alignItems: 'center', marginTop: 14 },
  segmentLabel: { width: 70, fontSize: 15, color: '#444' },
  segment: { flex: 1, flexDirection: 'row', backgroundColor: '#e4e6ea', borderRadius: 10, padding: 3 },
  segmentItem: { flex: 1, paddingVertical: 8, borderRadius: 8, alignItems: 'center' },
  segmentItemActive: { backgroundColor: '#fff' },
  segmentText: { color: '#555', fontSize: 14 },
  segmentTextActive: { color: '#111', fontWeight: '700' },
  error: { color: '#ff5c5c', marginTop: 12, fontSize: 14, fontWeight: '600' },

  camera: { flex: 1, backgroundColor: '#000' },
  hud: { position: 'absolute', top: 0, left: 0, right: 0, paddingHorizontal: 16, backgroundColor: 'rgba(0,0,0,0.35)', paddingBottom: 10 },
  hudTitle: { color: '#fff', fontSize: 18, fontWeight: '700', ...shadow },
  hudBig: { fontSize: 34, fontWeight: '800' },
  hudDim: { color: '#ccc', fontSize: 15, fontWeight: '400' },
  hudText: { color: '#fff', fontSize: 16, marginTop: 2, ...shadow },
  cue: { color: '#ffc53d', fontSize: 16, fontWeight: '600', marginTop: 4, ...shadow },
  notice: { color: '#ff9f40', fontSize: 15, fontWeight: '600', marginTop: 4, ...shadow },
  bottom: { position: 'absolute', left: 0, right: 0, bottom: 0, paddingHorizontal: 16, alignItems: 'center' },
  status: {
    color: '#fff', backgroundColor: 'rgba(220,40,40,0.85)', fontSize: 17, fontWeight: '700',
    paddingHorizontal: 14, paddingVertical: 8, borderRadius: 10, overflow: 'hidden', marginBottom: 10,
  },
  perf: { color: '#ddd', fontSize: 13, marginBottom: 10, ...shadow },
  buttons: { flexDirection: 'row', gap: 10 },
  buttonsLight: { flexDirection: 'row', gap: 10, marginTop: 20, paddingHorizontal: 16 },
  button: { backgroundColor: 'rgba(255,255,255,0.2)', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 12 },
  buttonText: { color: '#fff', fontSize: 15, fontWeight: '700' },
  buttonPrimary: { backgroundColor: '#fff' },
  buttonPrimaryText: { color: '#111' },
  buttonOutline: { backgroundColor: 'transparent', borderWidth: 1, borderColor: '#bbb' },

  summary: { flex: 1, backgroundColor: '#f6f7f9' },
  statRow: { flexDirection: 'row', gap: 10, paddingHorizontal: 16, marginTop: 16 },
  stat: { flex: 1, backgroundColor: '#fff', borderRadius: 14, padding: 14, alignItems: 'center', elevation: 1 },
  statValue: { fontSize: 28, fontWeight: '800', color: '#111' },
  statLabel: { fontSize: 13, color: '#666', marginTop: 2 },
  tableWrap: { marginTop: 20, marginHorizontal: 16, backgroundColor: '#fff', borderRadius: 14 },
  tr: { flexDirection: 'row', borderBottomWidth: StyleSheet.hairlineWidth, borderColor: '#ddd' },
  th: { backgroundColor: '#eef0f3' },
  thText: { fontWeight: '700', color: '#333' },
  td: { width: 68, paddingVertical: 8, paddingHorizontal: 6, fontSize: 13, color: '#222' },
  tdWide: { width: 260 },
  empty: { padding: 16, color: '#666' },
  perfSummary: { marginTop: 16, paddingHorizontal: 20, color: '#555', fontSize: 13, lineHeight: 20 },
});
