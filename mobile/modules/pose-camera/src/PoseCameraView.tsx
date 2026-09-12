import { requireNativeView } from 'expo';
import * as React from 'react';

import { PoseCameraViewProps } from './PoseCamera.types';

const NativeView: React.ComponentType<PoseCameraViewProps> = requireNativeView('PoseCamera');

export default function PoseCameraView(props: PoseCameraViewProps) {
  return <NativeView {...props} />;
}
