package expo.modules.posecamera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import expo.modules.kotlin.AppContext
import expo.modules.kotlin.viewevent.EventDispatcher
import expo.modules.kotlin.views.ExpoView
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

private const val TAG = "PoseCamera"
private const val STUCK_FRAME_NS = 1_000_000_000L

/**
 * CameraX preview + BlazePose (MediaPipe PoseLandmarker, LIVE_STREAM mode).
 *
 * Frames are analysed on [analysisExecutor], one at a time: while a frame is inside the
 * landmarker, new camera frames are dropped, so `inferenceMs` is pure model latency and the
 * event rate is the real pose FPS. Landmarks are in the (rotated, un-mirrored) camera image;
 * `mirrored` tells JS to flip x for display when the front camera preview is mirrored.
 */
class PoseCameraView(context: Context, appContext: AppContext) : ExpoView(context, appContext) {
  // React Native doesn't lay out native children; PreviewView needs a real measure/layout pass.
  override val shouldUseAndroidLayout = true

  private val onPose by EventDispatcher()
  private val onError by EventDispatcher()

  var model = "full"
  var cameraFacing = "front"
  private var boundModel: String? = null
  private var boundFacing: String? = null

  private val previewView = PreviewView(context).apply {
    layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT)
    scaleType = PreviewView.ScaleType.FILL_CENTER
    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
  }

  private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
  private var cameraProvider: ProcessCameraProvider? = null

  // Owned by analysisExecutor.
  private var landmarker: PoseLandmarker? = null
  private var lastTimestampMs = -1L

  @Volatile private var delegate = "GPU"
  @Volatile private var inFlightStartNs = 0L
  @Volatile private var preprocessMs = 0.0
  @Volatile private var imageWidth = 0
  @Volatile private var imageHeight = 0
  @Volatile private var mirrored = true
  @Volatile private var released = false

  init {
    addView(previewView)
  }

  fun applyProps() {
    if (released) return
    if (model != boundModel) {
      val variant = model
      boundModel = variant
      analysisExecutor.execute { createLandmarker(variant, preferGpu = true) }
    }
    if (cameraFacing != boundFacing) {
      boundFacing = cameraFacing
      bindCamera(front = cameraFacing != "back")
    }
  }

  fun release() {
    if (released) return
    released = true
    try {
      cameraProvider?.unbindAll()
    } catch (e: Exception) {
      Log.w(TAG, "unbind failed", e)
    }
    analysisExecutor.execute {
      landmarker?.close()
      landmarker = null
    }
    analysisExecutor.shutdown()
  }

  // --- MediaPipe ------------------------------------------------------------

  private fun createLandmarker(variant: String, preferGpu: Boolean) {
    if (released) return
    landmarker?.close()
    landmarker = null
    inFlightStartNs = 0L
    val buffer = try {
      loadModel(variant)
    } catch (e: Exception) {
      emitError("Could not load pose_landmarker_$variant.task: ${e.message}")
      return
    }
    val delegates = if (preferGpu) listOf(Delegate.GPU, Delegate.CPU) else listOf(Delegate.CPU)
    for (d in delegates) {
      try {
        buffer.rewind()
        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
          .setBaseOptions(BaseOptions.builder().setModelAssetBuffer(buffer).setDelegate(d).build())
          .setRunningMode(RunningMode.LIVE_STREAM)
          .setNumPoses(1)
          .setResultListener { result, _ -> onResult(result) }
          .setErrorListener { e -> onLandmarkerError(e) }
          .build()
        landmarker = PoseLandmarker.createFromOptions(context, options)
        delegate = d.name
        Log.i(TAG, "PoseLandmarker ready: model=$variant delegate=${d.name}")
        return
      } catch (e: Exception) {
        Log.w(TAG, "PoseLandmarker init with ${d.name} failed", e)
      }
    }
    emitError("Could not create the pose landmarker")
  }

  private fun loadModel(variant: String): ByteBuffer {
    val bytes = context.assets.open("pose_landmarker_$variant.task").use { it.readBytes() }
    return ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
      put(bytes)
      rewind()
    }
  }

  private fun analyze(image: ImageProxy) {
    val lm = landmarker
    val busySince = inFlightStartNs
    val now = SystemClock.elapsedRealtimeNanos()
    if (lm == null || released || (busySince != 0L && now - busySince < STUCK_FRAME_NS)) {
      image.close()
      return
    }
    val bitmap: Bitmap
    try {
      val raw = image.toBitmap()
      val rotation = image.imageInfo.rotationDegrees
      bitmap = if (rotation == 0) raw else Bitmap.createBitmap(
        raw, 0, 0, raw.width, raw.height, Matrix().apply { postRotate(rotation.toFloat()) }, false
      )
    } catch (e: Exception) {
      Log.w(TAG, "frame conversion failed", e)
      return
    } finally {
      image.close()
    }
    imageWidth = bitmap.width
    imageHeight = bitmap.height

    // LIVE_STREAM needs strictly increasing timestamps.
    val start = SystemClock.elapsedRealtimeNanos()
    preprocessMs = (start - now) / 1e6
    val ts = maxOf(start / 1_000_000, lastTimestampMs + 1)
    lastTimestampMs = ts
    inFlightStartNs = start
    try {
      lm.detectAsync(BitmapImageBuilder(bitmap).build(), ts)
    } catch (e: Exception) {
      inFlightStartNs = 0L
      Log.w(TAG, "detectAsync failed", e)
    }
  }

  private fun onResult(result: PoseLandmarkerResult) {
    val start = inFlightStartNs
    val inferenceMs = if (start != 0L) (SystemClock.elapsedRealtimeNanos() - start) / 1e6 else 0.0
    inFlightStartNs = 0L

    val landmarks = result.landmarks().firstOrNull()?.map {
      mapOf(
        "x" to it.x().toDouble(),
        "y" to it.y().toDouble(),
        "z" to it.z().toDouble(),
        "visibility" to it.visibility().orElse(0f).toDouble(),
      )
    } ?: emptyList()
    val world = result.worldLandmarks().firstOrNull()?.map {
      mapOf("x" to it.x().toDouble(), "y" to it.y().toDouble(), "z" to it.z().toDouble())
    } ?: emptyList()

    val payload: Map<String, Any> = mapOf(
      "landmarks" to landmarks,
      "worldLandmarks" to world,
      "imageWidth" to imageWidth,
      "imageHeight" to imageHeight,
      "inferenceMs" to inferenceMs,
      "preprocessMs" to preprocessMs,
      "timestampMs" to result.timestampMs().toDouble(),
      "mirrored" to mirrored,
      "delegate" to delegate,
      "model" to (boundModel ?: model),
    )
    post { if (!released) onPose(payload) }
  }

  private fun onLandmarkerError(e: RuntimeException) {
    Log.e(TAG, "PoseLandmarker error (delegate=$delegate)", e)
    inFlightStartNs = 0L
    if (delegate == "GPU" && !released) {
      delegate = "CPU"
      emitError("GPU delegate failed, falling back to CPU: ${e.message}")
      val variant = boundModel ?: model
      try {
        analysisExecutor.execute { createLandmarker(variant, preferGpu = false) }
      } catch (_: Exception) {
        // executor already shut down
      }
    } else {
      emitError("Pose detection error: ${e.message}")
    }
  }

  // --- CameraX --------------------------------------------------------------

  private fun bindCamera(front: Boolean) {
    val owner = appContext.currentActivity as? LifecycleOwner
    if (owner == null) {
      emitError("Camera needs a LifecycleOwner activity")
      return
    }
    val future = ProcessCameraProvider.getInstance(context)
    future.addListener({
      if (released) return@addListener
      val provider = try {
        future.get()
      } catch (e: Exception) {
        emitError("Camera unavailable: ${e.message}")
        return@addListener
      }
      cameraProvider = provider

      // Same 4:3 aspect for preview and analysis so JS can map landmarks with "cover" math.
      val aspect = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
      val preview = Preview.Builder()
        .setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(aspect).build())
        .build()
      preview.setSurfaceProvider(previewView.surfaceProvider)

      val analysis = ImageAnalysis.Builder()
        .setResolutionSelector(
          ResolutionSelector.Builder()
            .setAspectRatioStrategy(aspect)
            .setResolutionStrategy(
              ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
            )
            .build()
        )
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        .build()
      analysis.setAnalyzer(analysisExecutor, ::analyze)

      val selector = if (front) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
      try {
        provider.unbindAll()
        mirrored = front
        provider.bindToLifecycle(owner, selector, preview, analysis)
      } catch (e: Exception) {
        emitError("Could not start the ${if (front) "front" else "back"} camera: ${e.message}")
      }
    }, ContextCompat.getMainExecutor(context))
  }

  private fun emitError(message: String) {
    Log.e(TAG, message)
    post { if (!released) onError(mapOf("message" to message)) }
  }
}
