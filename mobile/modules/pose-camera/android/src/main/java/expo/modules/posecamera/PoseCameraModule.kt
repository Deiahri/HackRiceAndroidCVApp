package expo.modules.posecamera

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class PoseCameraModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("PoseCamera")

    View(PoseCameraView::class) {
      Events("onPose", "onError")

      Prop("model") { view: PoseCameraView, value: String? ->
        view.model = if (value == "lite") "lite" else "full"
      }

      Prop("cameraFacing") { view: PoseCameraView, value: String? ->
        view.cameraFacing = if (value == "back") "back" else "front"
      }

      OnViewDidUpdateProps { view: PoseCameraView ->
        view.applyProps()
      }

      OnViewDestroys { view: PoseCameraView ->
        view.release()
      }
    }
  }
}
