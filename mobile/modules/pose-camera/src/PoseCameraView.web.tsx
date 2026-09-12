import { PoseCameraViewProps } from './PoseCamera.types';

// PoseCameraView is not available on the web platform.
export default function PoseCameraView(_props: PoseCameraViewProps) {
  throw new Error('PoseCameraView is not available on the web platform.');
}
