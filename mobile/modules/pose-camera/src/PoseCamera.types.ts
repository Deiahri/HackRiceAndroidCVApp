import type { StyleProp, ViewStyle } from 'react-native';

/** Normalized to the rotated, un-mirrored camera image: x, y in 0..1, z relative depth. */
export type Landmark = { x: number; y: number; z: number; visibility: number };

/** Metres, hip-centred (BlazePose world coordinates). */
export type WorldLandmark = { x: number; y: number; z: number };

export type PoseEventPayload = {
  /** 33 BlazePose landmarks, or empty when no person was detected in the frame. */
  landmarks: Landmark[];
  worldLandmarks: WorldLandmark[];
  imageWidth: number;
  imageHeight: number;
  /** detectAsync -> result latency for this frame. */
  inferenceMs: number;
  /** ImageProxy -> rotated Bitmap conversion time. */
  preprocessMs: number;
  timestampMs: number;
  /** True when the preview is mirrored (front camera): flip x to draw over it. */
  mirrored: boolean;
  delegate: 'GPU' | 'CPU';
  model: 'lite' | 'full';
};

export type ErrorEventPayload = { message: string };

export type PoseCameraViewProps = {
  model?: 'lite' | 'full';
  cameraFacing?: 'front' | 'back';
  onPose?: (event: { nativeEvent: PoseEventPayload }) => void;
  onError?: (event: { nativeEvent: ErrorEventPayload }) => void;
  style?: StyleProp<ViewStyle>;
};
