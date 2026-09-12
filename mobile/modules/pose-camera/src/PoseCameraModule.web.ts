import { registerWebModule, NativeModule } from 'expo';

// PoseCameraModule is not available on the web platform.
class PoseCameraModule extends NativeModule<{}> {}

export default registerWebModule(PoseCameraModule, 'PoseCameraModule');
