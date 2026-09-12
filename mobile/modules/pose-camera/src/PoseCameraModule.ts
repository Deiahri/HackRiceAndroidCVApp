import { NativeModule, requireNativeModule } from 'expo';

declare class PoseCameraModule extends NativeModule<{}> {}

export default requireNativeModule<PoseCameraModule>('PoseCamera');
