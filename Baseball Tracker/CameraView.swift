//
//  CameraView.swift
//  Baseball Tracker
//

import SwiftUI
import AVFoundation
import Vision
import CoreML
import Photos

struct CameraView: UIViewControllerRepresentable {
    var onVideoSaved: ((String) -> Void)? = nil
    var videoURL: URL? = nil
    @Binding var isRecordingLive: Bool
    @Environment(\.presentationMode) var presentationMode

    func makeUIViewController(context: Context) -> CameraViewController {
        let vc = CameraViewController()
        vc.onVideoSaved = onVideoSaved
        vc.videoURL = videoURL
        vc.onRecordingStateChanged = { value in
            DispatchQueue.main.async { self.isRecordingLive = value }
        }
        return vc
    }

    func updateUIViewController(_ uiViewController: CameraViewController, context: Context) {}
}

// MARK: - Parabolic Trajectory Smoother
class TrajectoryKalmanSmoother {
    
    static func smoothTrajectory(_ points: [CGPoint]) -> [CGPoint] {
        guard points.count >= 3 else { return points }
        
        let parabola = fitParabola(to: points)
        
        guard let (a, b, c) = parabola else {
            return points
        }
        
        let xValues = points.map { $0.x }
        guard let minX = xValues.min(), let maxX = xValues.max() else { return points }
        
        let numPoints = max(points.count * 4, 50)
        var smoothedPoints: [CGPoint] = []
        
        for i in 0..<numPoints {
            let t = Double(i) / Double(numPoints - 1)
            let x = Double(minX) + t * Double(maxX - minX)
            let y = a * x * x + b * x + c
            
            smoothedPoints.append(CGPoint(x: x, y: y))
        }
        
        return smoothedPoints
    }
    
    // MARK: - Parabola Fitting using Least Squares

    private static func fitParabola(to points: [CGPoint]) -> (a: Double, b: Double, c: Double)? {
        guard points.count >= 3 else { return nil }
        
        let n = Double(points.count)
        
        var sumX = 0.0, sumX2 = 0.0, sumX3 = 0.0, sumX4 = 0.0
        var sumY = 0.0, sumXY = 0.0, sumX2Y = 0.0
        
        for point in points {
            let x = Double(point.x)
            let y = Double(point.y)
            let x2 = x * x
            let x3 = x2 * x
            let x4 = x2 * x2
            
            sumX += x
            sumX2 += x2
            sumX3 += x3
            sumX4 += x4
            sumY += y
            sumXY += x * y
            sumX2Y += x2 * y
        }
        
        
        let matrix: [[Double]] = [
            [sumX4, sumX3, sumX2],
            [sumX3, sumX2, sumX],
            [sumX2, sumX, n]
        ]
        
        let vector = [sumX2Y, sumXY, sumY]
        
        guard let solution = solveLinearSystem(matrix: matrix, vector: vector) else {
            return nil
        }
        
        return (a: solution[0], b: solution[1], c: solution[2])
    }
    
    // MARK: - Linear System Solver (Gaussian Elimination)
    
    private static func solveLinearSystem(matrix: [[Double]], vector: [Double]) -> [Double]? {
        guard matrix.count == 3, matrix[0].count == 3, vector.count == 3 else { return nil }
        
        var aug = matrix.map { $0 }
        for i in 0..<3 {
            aug[i].append(vector[i])
        }
        
        for col in 0..<3 {
            var maxRow = col
            for row in (col + 1)..<3 {
                if abs(aug[row][col]) > abs(aug[maxRow][col]) {
                    maxRow = row
                }
            }
            
            if maxRow != col {
                aug.swapAt(col, maxRow)
            }
            
            if abs(aug[col][col]) < 1e-10 {
                return nil
            }
            
            for row in (col + 1)..<3 {
                let factor = aug[row][col] / aug[col][col]
                for j in col..<4 {
                    aug[row][j] -= factor * aug[col][j]
                }
            }
        }
        
        var solution = [Double](repeating: 0.0, count: 3)
        for i in (0..<3).reversed() {
            var sum = aug[i][3]
            for j in (i + 1)..<3 {
                sum -= aug[i][j] * solution[j]
            }
            solution[i] = sum / aug[i][i]
        }
        
        return solution
    }
}

class CameraViewController: UIViewController {
    
    // MARK: - Camera
    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private let visionQueue = DispatchQueue(label: "camera.vision.queue", qos: .userInteractive)
    
    private var videoDeviceInput: AVCaptureDeviceInput!
    private var videoOutput: AVCaptureVideoDataOutput!
    private var previewLayer: CALayer!
    private var player: AVPlayer?
    
    // MARK: - Recording
    private var recordButton: UIButton!
    private var isRecording = false
    
    var assetWriter: AVAssetWriter?
    var assetWriterInput: AVAssetWriterInput?
    var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    var recordingStartTime: CMTime?
    
    private var lastFrameTime: CMTime?
    private var frameCount: Int = 0
    
    private var activeFrameRate: Double = 30
    private var activeDimensions: CMVideoDimensions = CMVideoDimensions(width: 1920, height: 1080)
    
    // MARK: - CoreML / Vision
    private var visionModel: VNCoreMLModel!
    private var requests = [VNRequest]()
    private var lastObservations: [VNRecognizedObjectObservation] = []
    private let allowedClasses = ["baseball"]
    private let confidenceThreshold: Float = 0.35
    
    // MARK: - Overlay
    private var overlayLayer = CALayer()
    
    private var detectionMarkers: [CGPoint] = []
    
    // MARK: - UI
    private var processingOverlay: UIView?
    private var processingLabel: UILabel?
    private var processingSpinner: UIActivityIndicatorView?
    
    private var backButton: UIButton!
    
    var videoURL: URL? = nil
    var onVideoSaved: ((String) -> Void)?
    var onRecordingStateChanged: ((Bool) -> Void)?
    
    // MARK: - CIContext
    private let ciContext = CIContext()
    
    // MARK: - Orientation Flags
    private var mirrored = false
    private var contentFlippedVertically = false
    private var contentUpsideDown = false
    
    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        print("🎥 CameraViewController viewDidLoad started")
        view.backgroundColor = .black
        onRecordingStateChanged?(false)
        
        setupOverlay()
        setupModel()
        setupBackButton()
        setupRecordButton()
        
        self.detectionMarkers.removeAll()
        
        if let url = videoURL {
            print("🎥 Processing video file mode")
            processVideoFile(url)
        } else {
            print("🎥 Live camera mode - setting up camera...")
            sessionQueue.async { [weak self] in
                self?.setupCamera()
            }
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
        overlayLayer.frame = view.bounds
        overlayLayer.position = CGPoint(x: view.bounds.midX, y: view.bounds.midY)
        processingOverlay?.frame = view.bounds
        
        if let backButton = backButton {
            let safeArea = view.safeAreaInsets
            let buttonHeight: CGFloat = 44
            let buttonWidth: CGFloat = 70
            backButton.frame = CGRect(x: 16, y: safeArea.top + 8, width: buttonWidth, height: buttonHeight)
            view.bringSubviewToFront(backButton)
        }
        
        if let recordButton = recordButton {
            let safeArea = view.safeAreaInsets
            let buttonSize: CGFloat = 70
            recordButton.frame = CGRect(x: (view.bounds.width - buttonSize) / 2,
                                        y: view.bounds.height - safeArea.bottom - buttonSize - 16,
                                        width: buttonSize,
                                        height: buttonSize)
            view.bringSubviewToFront(recordButton)
        }
    }
    
    // MARK: - Back Button Setup
    private func setupBackButton() {
        let button = UIButton(type: .system)
        button.setTitle("Back", for: .normal)
        button.setTitleColor(.white, for: .normal)
        button.titleLabel?.font = UIFont.systemFont(ofSize: 18, weight: .medium)
        button.backgroundColor = UIColor.black.withAlphaComponent(0.4)
        button.layer.cornerRadius = 8
        button.addTarget(self, action: #selector(backButtonTapped), for: .touchUpInside)
        button.accessibilityLabel = "Back"
        button.accessibilityHint = "Dismisses the camera view"
        button.translatesAutoresizingMaskIntoConstraints = false
        
        view.addSubview(button)
        self.backButton = button
        
        NSLayoutConstraint.activate([
            button.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 16),
            button.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 8),
            button.widthAnchor.constraint(equalToConstant: 70),
            button.heightAnchor.constraint(equalToConstant: 44)
        ])
    }
    
    @objc private func backButtonTapped() {
        if isRecording {
            stopCustomRecording()
        }
        sessionQueue.async { [weak self] in
            self?.captureSession.stopRunning()
        }
        self.dismiss(animated: true)
    }
    
    // MARK: - Record Button Setup
    private func setupRecordButton() {
        let button = UIButton(type: .custom)
        button.backgroundColor = UIColor.red.withAlphaComponent(0.7)
        button.layer.cornerRadius = 35
        button.layer.borderColor = UIColor.white.cgColor
        button.layer.borderWidth = 2
        button.addTarget(self, action: #selector(recordButtonTapped), for: .touchUpInside)
        button.accessibilityLabel = "Record"
        button.accessibilityHint = "Start or stop video recording"
        button.translatesAutoresizingMaskIntoConstraints = false
        
        view.addSubview(button)
        self.recordButton = button
        
        updateRecordButton(isRecording: false)
    }
    
    private func updateRecordButton(isRecording: Bool) {
        DispatchQueue.main.async {
            if isRecording {
                self.recordButton.backgroundColor = UIColor.red.withAlphaComponent(1.0)
                self.recordButton.layer.borderWidth = 0
            } else {
                self.recordButton.backgroundColor = UIColor.red.withAlphaComponent(0.7)
                self.recordButton.layer.borderWidth = 2
            }
        }
    }
    
    @objc private func recordButtonTapped() {
        if isRecording {
            stopCustomRecording()
        } else {
            startCustomRecording()
        }
    }
    
    private func startCustomRecording() {
        guard !isRecording else { return }
        
        let orientation = UIDevice.current.orientation
        let isPortrait = orientation == .portrait || orientation == .portraitUpsideDown
        let baseWidth = Int(self.activeDimensions.width)
        let baseHeight = Int(self.activeDimensions.height)
        let videoWidth = isPortrait ? min(baseWidth, baseHeight) : max(baseWidth, baseHeight)
        let videoHeight = isPortrait ? max(baseWidth, baseHeight) : min(baseWidth, baseHeight)
        let videoTransform = CGAffineTransform.identity
        
        let outputFileName = UUID().uuidString
        let outputFilePath = (NSTemporaryDirectory() as NSString).appendingPathComponent(outputFileName + ".mov")
        let outputURL = URL(fileURLWithPath: outputFilePath)
        
        do {
            assetWriter = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
        } catch {
            print("Failed to create AVAssetWriter: \(error)")
            return
        }
        
        print("[Recording] Using ProRes for \(videoWidth)x\(videoHeight) @ \(self.activeFrameRate) fps")
        
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.proRes422Proxy,
            AVVideoWidthKey: videoWidth,
            AVVideoHeightKey: videoHeight
        ]
        
        assetWriterInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        assetWriterInput?.expectsMediaDataInRealTime = true
        
        let mediaTimeScale = CMTimeScale(max(600, Int32(self.activeFrameRate * 10)))
        assetWriter?.movieTimeScale = mediaTimeScale
        assetWriterInput?.mediaTimeScale = mediaTimeScale
        
        assetWriterInput?.performsMultiPassEncodingIfSupported = false
        
        assetWriterInput?.transform = videoTransform
        
        guard let assetWriter = assetWriter,
              let assetWriterInput = assetWriterInput else {
            print("Asset writer or input not available")
            return
        }
        
        if assetWriter.canAdd(assetWriterInput) {
            assetWriter.add(assetWriterInput)
        } else {
            print("Cannot add asset writer input")
            return
        }
        
        let sourcePixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
            kCVPixelBufferWidthKey as String: videoWidth,
            kCVPixelBufferHeightKey as String: videoHeight,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        
        pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: assetWriterInput,
                                                                  sourcePixelBufferAttributes: sourcePixelBufferAttributes)
        
        recordingStartTime = nil
        
        self.detectionMarkers.removeAll()
        
        // Reset frame timing debug
        self.lastFrameTime = nil
        self.frameCount = 0
        
        isRecording = true
        updateRecordButton(isRecording: true)
        onRecordingStateChanged?(true)
        
        print("[Recording] Started recording at \(self.activeFrameRate) fps using ProRes 422 Proxy")
    }
    
    private func stopCustomRecording() {
        guard isRecording, let assetWriter = assetWriter, let assetWriterInput = assetWriterInput else { return }
        
        isRecording = false
        updateRecordButton(isRecording: false)
        onRecordingStateChanged?(false)
        
        if let startTime = recordingStartTime, let endTime = lastFrameTime {
            let duration = CMTimeGetSeconds(CMTimeSubtract(endTime, startTime))
            let avgFPS = Double(frameCount) / duration
            print("[Recording] Stopped. Recorded \(frameCount) frames in \(String(format: "%.2f", duration))s = \(String(format: "%.1f", avgFPS)) fps average")
        }
        
        assetWriterInput.markAsFinished()
        assetWriter.finishWriting { [weak self] in
            guard let self = self else { return }
            let rawVideoURL = assetWriter.outputURL
            
            print("[Recording] Raw video saved to: \(rawVideoURL.path)")
            print("[Recording] Starting annotation post-processing...")
            
            DispatchQueue.main.async {
                self.showProcessingOverlay(message: "Adding annotations to video...")
            }
            
            Task {
                await self.postProcessVideoWithAnnotations(rawVideoURL: rawVideoURL)
            }
            
            self.assetWriter = nil
            self.assetWriterInput = nil
            self.pixelBufferAdaptor = nil
            self.recordingStartTime = nil
        }
    }
    
    private func dismissToRootAndPresentAlert(_ alert: UIAlertController) {
        DispatchQueue.main.async {
            let presentAlert = {
                if let root = self.view.window?.rootViewController {
                    var top = root
                    while let presented = top.presentedViewController { top = presented }
                    top.present(alert, animated: true)
                } else {
                    self.topMostViewController().present(alert, animated: true)
                }
            }
            if let nav = self.navigationController {
                nav.popToRootViewController(animated: true)
                nav.dismiss(animated: true) {
                    presentAlert()
                }
            } else if let presenting = self.presentingViewController {
                var rootPresenter = presenting
                while let p = rootPresenter.presentingViewController { rootPresenter = p }
                rootPresenter.presentedViewController?.dismiss(animated: true) {
                    presentAlert()
                }
            } else {
                self.view.window?.rootViewController?.dismiss(animated: true) {
                    presentAlert()
                }
            }
        }
    }
    
    private func saveToPhotosThenAlert(outputURL: URL) {
        Task { @MainActor in
            await withCheckedContinuation { continuation in
                self.ensurePhotoLibraryPermission { granted in
                    Task { @MainActor in
                        guard granted else {
                            let alert = UIAlertController(title: "Photos Access Denied",
                                                          message: "Annotated video exported to a temporary file, but app does not have permission to save to Photos. Please enable Photos access in Settings.",
                                                          preferredStyle: .alert)
                            alert.addAction(UIAlertAction(title: "OK", style: .default))
                            self.dismissToRootAndPresentAlert(alert)
                            continuation.resume()
                            return
                        }
                        var localIdentifier: String?
                        PHPhotoLibrary.shared().performChanges({
                            if let placeholder = PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: outputURL)?.placeholderForCreatedAsset {
                                localIdentifier = placeholder.localIdentifier
                            }
                        }) { success, error in
                            Task { @MainActor in
                                if success, let id = localIdentifier {
                                    self.onVideoSaved?(id)
                                    let alert = UIAlertController(title: "Success",
                                                                  message: "Annotated video saved to Photos.",
                                                                  preferredStyle: .alert)
                                    alert.addAction(UIAlertAction(title: "OK", style: .default))
                                    self.dismissToRootAndPresentAlert(alert)
                                } else {
                                    self.onVideoSaved?("")
                                    let alert = UIAlertController(title: "Save Failed",
                                                                  message: "Could not save video to Photos. Please check permissions or available space.",
                                                                  preferredStyle: .alert)
                                    alert.addAction(UIAlertAction(title: "OK", style: .default))
                                    self.dismissToRootAndPresentAlert(alert)
                                }
                                continuation.resume()
                            }
                        }
                    }
                }
            }
        }
    }
    
    // MARK: - Processing Overlay
    private func showProcessingOverlay(message: String) {
        if processingOverlay == nil {
            let overlay = UIView(frame: view.bounds)
            overlay.backgroundColor = UIColor.black.withAlphaComponent(0.4)
            
            let spinner = UIActivityIndicatorView(style: .large)
            spinner.translatesAutoresizingMaskIntoConstraints = false
            spinner.startAnimating()
            
            let label = UILabel()
            label.translatesAutoresizingMaskIntoConstraints = false
            label.textColor = .white
            label.font = UIFont.preferredFont(forTextStyle: .headline)
            label.textAlignment = .center
            label.numberOfLines = 0
            
            overlay.addSubview(spinner)
            overlay.addSubview(label)
            
            NSLayoutConstraint.activate([
                spinner.centerXAnchor.constraint(equalTo: overlay.centerXAnchor),
                spinner.centerYAnchor.constraint(equalTo: overlay.centerYAnchor, constant: -12),
                label.topAnchor.constraint(equalTo: spinner.bottomAnchor, constant: 12),
                label.leadingAnchor.constraint(equalTo: overlay.leadingAnchor, constant: 24),
                label.trailingAnchor.constraint(equalTo: overlay.trailingAnchor, constant: -24)
            ])
            
            processingOverlay = overlay
            processingLabel = label
            processingSpinner = spinner
            view.addSubview(overlay)
        }
        processingOverlay?.isHidden = false
        updateProcessingMessage(message)
    }
    
    private func hideProcessingOverlay() {
        processingOverlay?.isHidden = true
    }
    
    private func updateProcessingMessage(_ message: String) {
        processingLabel?.text = message
    }
    
    // MARK: - Camera Setup
    private func setupCamera() {
        print("🎥 setupCamera() called on thread: \(Thread.current)")
        captureSession.beginConfiguration()

        captureSession.sessionPreset = .inputPriority
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            print("❌ ERROR: Could not get camera device")
            return
        }
        
        print("✅ Got camera device: \(device.localizedName)")
        
        let preferredDims: [(width: Int32, height: Int32)] = [
            (1920, 1080), (1080, 1920),
            (1280, 720), (720, 1280)
        ]
        
        func maxFPS(for format: AVCaptureDevice.Format) -> Double {
            return format.videoSupportedFrameRateRanges.map { $0.maxFrameRate }.max() ?? 0
        }
        
        let globalMaxFPS = device.formats.map { maxFPS(for: $0) }.max() ?? 0
        print("[Camera] Found \(device.formats.count) formats, max fps available: \(globalMaxFPS)")
        
        let highestFPSFormats = device.formats.filter { maxFPS(for: $0) == globalMaxFPS }
        print("[Camera] \(highestFPSFormats.count) formats support \(globalMaxFPS) fps")
        
        let preferredHighSpeedFormat: AVCaptureDevice.Format? = {
            for dim in preferredDims {
                if let match = highestFPSFormats.first(where: { fmt in
                    let d = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
                    return (d.width == dim.width && d.height == dim.height)
                }) {
                    return match
                }
            }
            return nil
        }()
        
        let chosenFormat: AVCaptureDevice.Format? = preferredHighSpeedFormat ?? device.formats.max(by: { f1, f2 in
            let fps1 = maxFPS(for: f1)
            let fps2 = maxFPS(for: f2)
            if fps1 == fps2 {
                let d1 = CMVideoFormatDescriptionGetDimensions(f1.formatDescription)
                let d2 = CMVideoFormatDescriptionGetDimensions(f2.formatDescription)
                let area1 = Int(d1.width) * Int(d1.height)
                let area2 = Int(d2.width) * Int(d2.height)
                return area1 < area2
            }
            return fps1 < fps2
        })
        
        if let bestFormat = chosenFormat {
            let maxFrameRate = maxFPS(for: bestFormat)
            let dims = CMVideoFormatDescriptionGetDimensions(bestFormat.formatDescription)
            print("[Camera] Chose format: \(dims.width)x\(dims.height) supporting max \(maxFrameRate) fps")
            
            do {
                try device.lockForConfiguration()
                device.activeFormat = bestFormat
                let desc = CMVideoFormatDescriptionGetDimensions(bestFormat.formatDescription)
                self.activeDimensions = desc
                if let frameRateRange = bestFormat.videoSupportedFrameRateRanges.first(where: { $0.maxFrameRate == maxFrameRate }) {
                    let duration = CMTimeMake(value: 1, timescale: Int32(frameRateRange.maxFrameRate.rounded()))
                    device.activeVideoMinFrameDuration = duration
                    device.activeVideoMaxFrameDuration = duration
                    print("[Camera] Set frame duration to \(duration.value)/\(duration.timescale)")
                }
                device.unlockForConfiguration()
                
                let actualDuration = device.activeVideoMaxFrameDuration
                if actualDuration.seconds > 0 {
                    self.activeFrameRate = 1.0 / actualDuration.seconds
                } else {
                    self.activeFrameRate = Double(maxFrameRate)
                }
                
                print("[Camera] Active configuration: \(self.activeDimensions.width)x\(self.activeDimensions.height) @ \(self.activeFrameRate) fps")
            } catch {
                print("Failed to configure device for highest frame rate: \(error)")
            }
        } else {
            print("[Camera] ERROR: No suitable format found!")
        }
        
        do {
            videoDeviceInput = try AVCaptureDeviceInput(device: device)
        } catch {
            print("Error creating device input: \(error)")
            return
        }
        
        guard videoDeviceInput != nil else { return }
        captureSession.addInput(videoDeviceInput)
        
        videoOutput = AVCaptureVideoDataOutput()
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
        ]
        
        let videoQueue = DispatchQueue(label: "videoQueue", qos: .userInteractive, attributes: [], autoreleaseFrequency: .workItem)
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        videoOutput.alwaysDiscardsLateVideoFrames = false
        
        if let connection = videoOutput.connection(with: .video) {
            if #available(iOS 17.0, *) {
                let desiredAngle: CGFloat = 90
                if connection.isVideoRotationAngleSupported(desiredAngle) {
                    connection.videoRotationAngle = desiredAngle
                }
            } else {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }
            
            self.mirrored = (videoDeviceInput.device.position == .front)
            self.contentFlippedVertically = false
        }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let bounds = self.view.bounds
            let preview = AVCaptureVideoPreviewLayer(session: self.captureSession)
            preview.videoGravity = .resizeAspectFill
            preview.frame = bounds
            self.previewLayer = preview
            self.view.layer.insertSublayer(preview, at: 0)
        }
        
        self.sessionQueue.async { [weak self] in
            self?.captureSession.startRunning()
        }
        
        captureSession.commitConfiguration()
    }
    
    // MARK: - Overlay & CoreML
    private func setupOverlay() {
        overlayLayer.frame = view.bounds
        view.layer.addSublayer(overlayLayer)
    }
    
    private func setupModel() {
        guard let mlmodel = try? best(configuration: MLModelConfiguration()).model,
              let visionModel = try? VNCoreMLModel(for: mlmodel) else { fatalError("Could not load weights") }
        self.visionModel = visionModel
        
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, _ in
            guard let self = self else { return }
            let results = (request.results as? [VNRecognizedObjectObservation]) ?? []
            self.lastObservations = results
            
            let baseballDetections = results.filter { obs in
                guard let label = obs.labels.first else { return false }
                return label.identifier == "baseball" && obs.confidence >= self.confidenceThreshold
            }.sorted { $0.confidence > $1.confidence }
            
            DispatchQueue.main.async {
                self.overlayLayer.sublayers?.forEach { $0.removeFromSuperlayer() }
                
                if let bestDetection = baseballDetections.first {
                    self.drawBoundingBox(bestDetection.boundingBox,
                                         bufferSize: self.view.bounds.size,
                                         confidence: bestDetection.confidence,
                                         mirrored: self.mirrored,
                                         contentFlippedVertically: self.contentFlippedVertically)
                }
            }
        }
        request.imageCropAndScaleOption = .scaleFill
        self.requests = [request]
    }
    
    // MARK: - Coordinate Conversion
    private func convertBoundingBox(_ rect: CGRect, bufferSize: CGSize, mirrored: Bool, contentFlippedVertically: Bool) -> CGRect {
        var x = rect.origin.x * bufferSize.width
        var y = rect.origin.y * bufferSize.height
        let w = rect.width * bufferSize.width
        let h = rect.height * bufferSize.height

        if mirrored { x = bufferSize.width - x - w }
        if !contentFlippedVertically {
            y = bufferSize.height - y - h
        }

        return CGRect(x: x, y: y, width: w, height: h)
    }

    private func convertPoint(_ point: CGPoint, bufferSize: CGSize, mirrored: Bool, contentFlippedVertically: Bool) -> CGPoint {
        var x = point.x * bufferSize.width
        var y = point.y * bufferSize.height

        if mirrored { x = bufferSize.width - x }
        if !contentFlippedVertically {
            y = bufferSize.height - y
        }

        return CGPoint(x: x, y: y)
    }
    
    private func drawBoundingBox(_ rect: CGRect, bufferSize: CGSize,
                                 confidence: VNConfidence, mirrored: Bool,
                                 contentFlippedVertically: Bool) {
        let boxRect = convertBoundingBox(rect, bufferSize: bufferSize,
                                         mirrored: mirrored, contentFlippedVertically: contentFlippedVertically)
        
        let centerX = boxRect.midX
        let centerY = boxRect.midY
        let radius = max(boxRect.width, boxRect.height) / 2
        
        let circleLayer = CAShapeLayer()
        let circlePath = UIBezierPath(ovalIn: CGRect(x: boxRect.minX, 
                                                      y: boxRect.minY,
                                                      width: boxRect.width,
                                                      height: boxRect.height))
        circleLayer.path = circlePath.cgPath
        circleLayer.strokeColor = UIColor.red.cgColor
        circleLayer.fillColor = UIColor.clear.cgColor
        circleLayer.lineWidth = 3
        
        let textLayer = CATextLayer()
        textLayer.string = String(format: "%.0f%%", confidence * 100)
        textLayer.foregroundColor = UIColor.red.cgColor
        textLayer.backgroundColor = UIColor.clear.cgColor
        textLayer.fontSize = 28
        textLayer.alignmentMode = .center
        textLayer.contentsScale = view.traitCollection.displayScale
        
        let textWidth: CGFloat = 80
        let textHeight: CGFloat = 35
        textLayer.frame = CGRect(x: centerX - textWidth / 2,
                                y: centerY - textHeight / 2,
                                width: textWidth,
                                height: textHeight)
        
        overlayLayer.addSublayer(circleLayer)
        overlayLayer.addSublayer(textLayer)
    }
    
    // MARK: - Post-Process Video with Annotations
    private func postProcessVideoWithAnnotations(rawVideoURL: URL) async {
        print("[PostProcess] Loading video asset from: \(rawVideoURL.path)")
        
        let asset = AVURLAsset(url: rawVideoURL)
        
        do {
            let tracks = try await asset.loadTracks(withMediaType: .video)
            guard let track = tracks.first else {
                print("[PostProcess] ERROR: No video track found")
                await MainActor.run { self.hideProcessingOverlay() }
                return
            }
            
            self.detectionMarkers.removeAll()
            
            await self.exportAnnotatedVideo(asset: asset, track: track)
            
        } catch {
            print("[PostProcess] Error: \(error)")
            await MainActor.run {
                self.hideProcessingOverlay()
                self.saveToPhotosThenAlert(outputURL: rawVideoURL)
            }
        }
    }
    
    // MARK: - Export Annotated Video
    private func exportAnnotatedVideo(asset: AVAsset, track: AVAssetTrack) async {
        do {
            let trimDuration = CMTime(seconds: 0.5, preferredTimescale: 600)
            let duration = try await asset.load(.duration)
            let startTime = trimDuration
            let endTime = duration
            let timeRange = CMTimeRange(start: startTime, end: endTime)
            
            let reader = try AVAssetReader(asset: asset)
            reader.timeRange = timeRange
            let outputSettings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
            let readerOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
            reader.add(readerOutput)
            reader.startReading()

            let naturalSize = try await track.load(.naturalSize)
            let width = Int(abs(naturalSize.width))
            let height = Int(abs(naturalSize.height))

            let outputURL = FileManager.default.temporaryDirectory.appendingPathComponent("annotatedVideo_\(UUID().uuidString).mov")
            try? FileManager.default.removeItem(at: outputURL)

            guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mov) else { return }
            let videoSettings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
                AVVideoCompressionPropertiesKey: [
                    AVVideoExpectedSourceFrameRateKey: Int(self.activeFrameRate),
                    AVVideoAverageBitRateKey: 20_000_000,
                    AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
                ]
            ]
            let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
            writerInput.expectsMediaDataInRealTime = false
            
            let mediaTimeScale = CMTimeScale(max(600, Int32(self.activeFrameRate * 10)))
            writer.movieTimeScale = mediaTimeScale
            writerInput.mediaTimeScale = mediaTimeScale

            let preferredTransform = try await track.load(.preferredTransform)
            writerInput.transform = preferredTransform

            let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: writerInput,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
                    kCVPixelBufferWidthKey as String: width,
                    kCVPixelBufferHeightKey as String: height
                ]
            )

            guard writer.canAdd(writerInput) else { return }
            writer.add(writerInput)
            writer.startWriting()
            writer.startSession(atSourceTime: .zero)

            while reader.status == .reading {
                guard let sampleBuffer = readerOutput.copyNextSampleBuffer(),
                      let px = CMSampleBufferGetImageBuffer(sampleBuffer) else { break }

                autoreleasepool {
                    let req = VNCoreMLRequest(model: self.visionModel)
                    let handler = VNImageRequestHandler(cvPixelBuffer: px, options: [:])
                    try? handler.perform([req])
                    let results = (req.results as? [VNRecognizedObjectObservation]) ?? []
                    
                    let baseballDetections = results.filter { obs in
                        guard let label = obs.labels.first else { return false }
                        return label.identifier == "baseball" && obs.confidence >= self.confidenceThreshold
                    }.sorted { $0.confidence > $1.confidence }
                    for obs in baseballDetections {
                        let c = CGPoint(x: obs.boundingBox.midX, y: obs.boundingBox.midY)
                        self.detectionMarkers.append(c)
                    }

                    let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                    while !writerInput.isReadyForMoreMediaData { usleep(1000) }
                    if let annotated = self.makeAnnotatedPixelBuffer(from: px, observations: results, pool: adaptor.pixelBufferPool) {
                        _ = adaptor.append(annotated, withPresentationTime: pts)
                    } else {
                        _ = adaptor.append(px, withPresentationTime: pts)
                    }
                }
            }

            writerInput.markAsFinished()
            await writer.finishWriting()
            await self.saveAndShowExportCompletion(outputURL: outputURL)

        } catch { print("Error exporting annotated video: \(error)") }
    }
    
    // MARK: - Permissions Helper
    private func ensurePhotoLibraryPermission(completion: @escaping (Bool) -> Void) {
        let status = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        switch status {
        case .authorized, .limited:
            completion(true)
        case .notDetermined:
            PHPhotoLibrary.requestAuthorization(for: .addOnly) { newStatus in
                DispatchQueue.main.async {
                    completion(newStatus == .authorized || newStatus == .limited)
                }
            }
        default:
            completion(false)
        }
    }

    // MARK: - Save to Photos and Show Completion for Exported Video

    @MainActor
    private func saveAndShowExportCompletion(outputURL: URL) async {
        await withCheckedContinuation { continuation in
            self.ensurePhotoLibraryPermission { granted in
                
                Task { @MainActor in
                    guard granted else {
                        self.onVideoSaved?("")
                        let alert = UIAlertController(title: "Photos Access Denied",
                                                      message: "Annotated video exported to a temporary file, but app does not have permission to save to Photos. Please enable Photos access in Settings.",
                                                      preferredStyle: .alert)
                        alert.addAction(UIAlertAction(title: "OK", style: .default) { _ in
                            self.hideProcessingOverlay()
                            self.dismiss(animated: true)
                        })
                        self.topMostViewController().present(alert, animated: true)
                        continuation.resume()
                        return
                    }
                    
                    var localIdentifier: String?
                    
                    PHPhotoLibrary.shared().performChanges({
                        if let placeholder = PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: outputURL)?.placeholderForCreatedAsset {
                            localIdentifier = placeholder.localIdentifier
                        }
                    }) { success, error in
                        Task { @MainActor in
                            if success, let id = localIdentifier {
                                self.onVideoSaved?(id)
                                let alert = UIAlertController(title: "Success",
                                                              message: "Annotated video saved to Photos.",
                                                              preferredStyle: .alert)
                                alert.addAction(UIAlertAction(title: "OK", style: .default) { _ in
                                    self.hideProcessingOverlay()
                                    self.dismiss(animated: true)
                                })
                                self.topMostViewController().present(alert, animated: true)
                            } else {
                                self.onVideoSaved?("")
                                let alert = UIAlertController(title: "Save Failed",
                                                              message: "Could not save video to Photos. Please check permissions or available space.",
                                                              preferredStyle: .alert)
                                alert.addAction(UIAlertAction(title: "OK", style: .default) { _ in
                                    self.hideProcessingOverlay()
                                    self.dismiss(animated: true)
                                })
                                self.topMostViewController().present(alert, animated: true)
                            }
                            continuation.resume()
                        }
                    }
                }
            }
        }
    }

    // MARK: - Annotation Renderer
    private func makeAnnotatedPixelBuffer(from pixelBuffer: CVPixelBuffer,
                                          observations: [VNRecognizedObjectObservation],
                                          pool: CVPixelBufferPool?) -> CVPixelBuffer? {
        guard let pool = pool else { return nil }

        var outPixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &outPixelBuffer)
        guard status == kCVReturnSuccess, let output = outPixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(output, [])
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        }

        let srcImage = CIImage(cvPixelBuffer: pixelBuffer)

        let rect = CGRect(x: 0, y: 0,
                          width: CVPixelBufferGetWidth(output),
                          height: CVPixelBufferGetHeight(output))

        UIGraphicsBeginImageContextWithOptions(rect.size, false, 1.0)
        guard let ctx = UIGraphicsGetCurrentContext() else {
            UIGraphicsEndImageContext()
            return output
        }

        let uiImage = UIImage(ciImage: srcImage)
        uiImage.draw(in: rect)

        let baseballDetections = observations.filter { obs in
            guard let label = obs.labels.first else { return false }
            return label.identifier == "baseball" && obs.confidence >= self.confidenceThreshold
        }.sorted { $0.confidence > $1.confidence }
        
        for obs in baseballDetections {
            let c = CGPoint(x: obs.boundingBox.midX, y: obs.boundingBox.midY)
            self.detectionMarkers.append(c)
        }
        
        if let obs = baseballDetections.first {
            let bbox = obs.boundingBox
            var x = bbox.origin.x * rect.width
            var y = bbox.origin.y * rect.height
            let w = bbox.width * rect.width
            let h = bbox.height * rect.height
            if !contentFlippedVertically {
                y = rect.height - y - h
            }
            
            if mirrored { x = rect.width - x - w }
            let drawRect = CGRect(x: x, y: y, width: w, height: h)

            ctx.setStrokeColor(UIColor.red.cgColor)
            ctx.setLineWidth(3)
            ctx.addEllipse(in: drawRect)
            ctx.strokePath()
            
            let centerX = drawRect.midX
            let centerY = drawRect.midY
            let confidenceText = String(format: "%.0f%%", obs.confidence * 100)
            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 28),
                .foregroundColor: UIColor.red
            ]
            
            ctx.saveGState()
            ctx.translateBy(x: centerX, y: centerY)
            ctx.rotate(by: -.pi / 2)
            
            let textSize = confidenceText.size(withAttributes: attributes)
            confidenceText.draw(at: CGPoint(x: -textSize.width / 2, y: -textSize.height / 2),
                               withAttributes: attributes)
            
            ctx.restoreGState()
        } else {
        }
        
        if self.detectionMarkers.count >= 3 {
            let smoothedPoints = TrajectoryKalmanSmoother.smoothTrajectory(self.detectionMarkers)
            
            ctx.setStrokeColor(UIColor.red.withAlphaComponent(0.9).cgColor)
            ctx.setLineWidth(25)
            ctx.setLineCap(.round)
            ctx.setLineJoin(.round)
            
            var sfx = smoothedPoints[0].x * rect.width
            var sfy = smoothedPoints[0].y * rect.height
            if !contentFlippedVertically { sfy = rect.height - sfy }
            if mirrored { sfx = rect.width - sfx }
            ctx.move(to: CGPoint(x: sfx, y: sfy))
            
            for i in 1..<smoothedPoints.count {
                var sx = smoothedPoints[i].x * rect.width
                var sy = smoothedPoints[i].y * rect.height
                if !contentFlippedVertically { sy = rect.height - sy }
                if mirrored { sx = rect.width - sx }
                ctx.addLine(to: CGPoint(x: sx, y: sy))
            }
            ctx.strokePath()
        }
        
        if !self.detectionMarkers.isEmpty {
            for normCenter in self.detectionMarkers {
                var cx = normCenter.x * rect.width
                var cy = normCenter.y * rect.height
                if !contentFlippedVertically { cy = rect.height - cy }
                if mirrored { cx = rect.width - cx }
                let markerSize: CGFloat = 12
                let markerRect = CGRect(x: cx - markerSize/2,
                                        y: cy - markerSize/2,
                                        width: markerSize,
                                        height: markerSize)
                ctx.setFillColor(UIColor.red.cgColor)
                ctx.fillEllipse(in: markerRect)
                ctx.setStrokeColor(UIColor.white.withAlphaComponent(0.8).cgColor)
                ctx.setLineWidth(2)
                ctx.strokeEllipse(in: markerRect)
            }
        }

        let composed = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        if let composed = composed, let cgImage = composed.cgImage {
            let ci = CIImage(cgImage: cgImage)
            ciContext.render(ci, to: output)
        }

        return output
    }

    // MARK: - Export Completion / Redirect
    private func showExportCompletion(url: URL) {
        DispatchQueue.main.async {
            self.onVideoSaved?(url.path)

            let alert = UIAlertController(title: "Success",
                                          message: "Recorded video saved to Photos.",
                                          preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "OK", style: .default) { _ in
                self.dismiss(animated: true)
            })
            self.present(alert, animated: true)
        }
    }

    // MARK: - Video File Processing
    private func processVideoFile(_ url: URL) {
        let asset = AVURLAsset(url: url)
        showProcessingOverlay(message: "Processing…")

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.player = AVPlayer(url: url)
            let playerLayer = AVPlayerLayer(player: self.player)
            playerLayer.frame = self.view.bounds
            playerLayer.videoGravity = .resizeAspectFill
            self.previewLayer = playerLayer
            self.view.layer.insertSublayer(self.previewLayer, at: 0)
            self.player?.play()
        }

        Task {
            do {
                let tracks = try await asset.loadTracks(withMediaType: .video)
                guard let track = tracks.first else { await MainActor.run { self.hideProcessingOverlay() }; return }

                let t = try await track.load(.preferredTransform)
                
                let angle = atan2(Double(t.b), Double(t.a))
                var degrees = Int(round(angle * 180.0 / .pi))
                degrees = (degrees % 360 + 360) % 360
                let snapped: Int = {
                    let options = [0, 90, 180, 270]
                    let diffs = options.map { abs($0 - degrees) }
                    if let idx = diffs.enumerated().min(by: { $0.element < $1.element })?.offset {
                        return options[idx]
                    }
                    return 0
                }()

                var mirrored = false
                switch snapped {
                case 0, 180:
                    mirrored = (t.a < 0)
                case 90, 270:
                    mirrored = (t.d < 0)
                default:
                    mirrored = false
                }
                self.mirrored = mirrored

                self.contentUpsideDown = (snapped == 180)

                self.contentFlippedVertically = false

                await MainActor.run { self.updateProcessingMessage("Analyzing & Exporting…") }
                self.processVideoAsset(asset, track: track)
                await self.exportAnnotatedVideo(asset: asset, track: track)
            } catch {
                await MainActor.run { self.hideProcessingOverlay() }
            }
        }
    }

    private func processVideoAsset(_ asset: AVAsset, track: AVAssetTrack) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let reader = try AVAssetReader(asset: asset)
                let outputSettings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
                let readerOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
                reader.add(readerOutput)
                reader.startReading()

                while reader.status == .reading {
                    if let sampleBuffer = readerOutput.copyNextSampleBuffer(),
                       let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
                        try? handler.perform(self.requests)
                        Thread.sleep(forTimeInterval: 0.03)
                    }
                }
            } catch { print("Error reading video: \(error)") }
        }
    }
}

// MARK: - Live Capture Delegate
extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard videoURL == nil else { return }

        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        
        if isRecording {
            guard let assetWriter = assetWriter,
                  let assetWriterInput = assetWriterInput,
                  let pixelBufferAdaptor = pixelBufferAdaptor,
                  let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            
            if recordingStartTime == nil {
                assetWriter.startWriting()
                assetWriter.startSession(atSourceTime: pts)
                recordingStartTime = pts
                print("[Recording] Writer started. Status: \(assetWriter.status.rawValue)")
            }
            
            if assetWriter.status == .failed {
                print("[Recording] ERROR: Writer failed with error: \(assetWriter.error?.localizedDescription ?? "unknown")")
                return
            }
            
            frameCount += 1
            if let lastTime = lastFrameTime {
                let deltaTime = CMTimeGetSeconds(CMTimeSubtract(pts, lastTime))
                let instantFPS = 1.0 / deltaTime
                
                if frameCount % 60 == 0 {
                    let avgTime = CMTimeGetSeconds(CMTimeSubtract(pts, recordingStartTime ?? pts))
                    let avgFPS = avgTime > 0 ? Double(frameCount) / avgTime : 0
                    print("[Recording] Frame \(frameCount): instant=\(String(format: "%.1f", instantFPS)) fps, avg=\(String(format: "%.1f", avgFPS)) fps, ready=\(assetWriterInput.isReadyForMoreMediaData)")
                }
            }
            lastFrameTime = pts
            
            if assetWriterInput.isReadyForMoreMediaData {
                let success = pixelBufferAdaptor.append(pixelBuffer, withPresentationTime: pts)
                if !success {
                    print("[Recording] WARNING: Failed to append frame \(frameCount)")
                }
            } else {
                print("[Recording] WARNING: Writer not ready, dropping frame \(frameCount)")
            }
            
            if frameCount % 4 == 0 {
                visionQueue.async { [weak self, sampleBuffer] in
                    guard let self = self,
                          let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
                    
                    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
                    try? handler.perform(self.requests)
                }
            }
            
            return
        }
        
        guard frameCount % 4 == 0 else {
            frameCount += 1
            return
        }
        frameCount += 1
        
        visionQueue.async { [weak self, sampleBuffer] in
            guard let self = self,
                  let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            try? handler.perform(self.requests)
        }
    }
}

// MARK: - UIViewController extension for safe alert presentation
private extension UIViewController {
    func topMostViewController() -> UIViewController {
        var top = self
        while let presented = top.presentedViewController {
            top = presented
        }
        return top
    }
}
