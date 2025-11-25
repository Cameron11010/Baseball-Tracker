//
//  CameraView.swift
//  Baseball Tracker
//
//  Supports live camera + processing of existing video files with YOLO tracking
//  iOS 26 safe
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

class CameraViewController: UIViewController {
    
    // MARK: - Camera
    private let captureSession = AVCaptureSession()
    // Serial queue for all capture session work to avoid blocking main thread
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    
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
    
    // Store the active camera frame rate (fps) and active dimensions; used to configure writer so Photos recognizes the asset as genuine high‑fps (slow‑motion) video
    private var activeFrameRate: Double = 30
    private var activeDimensions: CMVideoDimensions = CMVideoDimensions(width: 1920, height: 1080)
    
    // MARK: - CoreML / Vision
    private var visionModel: VNCoreMLModel!
    private var requests = [VNRequest]()
    private var lastObservations: [VNRecognizedObjectObservation] = []
    private let allowedClasses = ["baseball"]
    
    // MARK: - Overlay
    private var overlayLayer = CALayer()
    private var ballTrailNormalized: [CGPoint] = []
    private let maxTrailLength = 15
    
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
        view.backgroundColor = .black
        onRecordingStateChanged?(false)
        
        setupOverlay()
        setupModel()
        setupBackButton()
        setupRecordButton()
        
        if let url = videoURL {
            processVideoFile(url)
        } else {
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
        
        // Layout back button at top-left safe area
        if let backButton = backButton {
            let safeArea = view.safeAreaInsets
            let buttonHeight: CGFloat = 44
            let buttonWidth: CGFloat = 70
            backButton.frame = CGRect(x: 16, y: safeArea.top + 8, width: buttonWidth, height: buttonHeight)
            view.bringSubviewToFront(backButton)
        }
        
        // Layout record button at bottom center safe area
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
        
        // Constraints for accessibility even if frame is set in layoutSubviews
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
        
        // Detect device orientation and choose encoding dimensions based on active camera format
        let orientation = UIDevice.current.orientation
        let isPortrait = orientation == .portrait || orientation == .portraitUpsideDown
        // Use the active capture format dimensions to preserve high-fps format resolution, then orient
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
        
        // IMPORTANT: Include AVVideoExpectedSourceFrameRateKey with the active camera frame rate
        // This ensures the output video preserves the high frame rate chosen by the camera (e.g., 120 or 240 fps)
        // so that slow motion video plays natively in Photos and other players.
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: videoWidth,
            AVVideoHeightKey: videoHeight,
            AVVideoCompressionPropertiesKey: [
                AVVideoExpectedSourceFrameRateKey: Int(self.activeFrameRate),
                AVVideoAverageBitRateKey: 20_000_000,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
            ]
        ]
        
        assetWriterInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        assetWriterInput?.expectsMediaDataInRealTime = true
        
        // Preserve timing so the nominalFrameRate remains the high fps in the saved asset
        let mediaTimeScale = CMTimeScale(max(600, Int32(self.activeFrameRate * 10)))
        assetWriter?.movieTimeScale = mediaTimeScale
        assetWriterInput?.mediaTimeScale = mediaTimeScale
        
        // IMPORTANT: Do NOT apply rotation transform here.
        // Setting transform to .identity ensures the video is recorded upright.
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
        
        // Use width/height accordingly for pixel buffer attributes
        let sourcePixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: videoWidth,
            kCVPixelBufferHeightKey as String: videoHeight
        ]
        
        pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: assetWriterInput,
                                                                  sourcePixelBufferAttributes: sourcePixelBufferAttributes)
        
        recordingStartTime = nil
        
        isRecording = true
        updateRecordButton(isRecording: true)
        onRecordingStateChanged?(true)
    }
    
    private func stopCustomRecording() {
        guard isRecording, let assetWriter = assetWriter, let assetWriterInput = assetWriterInput else { return }
        
        isRecording = false
        updateRecordButton(isRecording: false)
        onRecordingStateChanged?(false)
        
        assetWriterInput.markAsFinished()
        assetWriter.finishWriting { [weak self] in
            guard let self = self else { return }
            let outputURL = assetWriter.outputURL
            
            // Use new saveToPhotosThenAlert method to save and then dismiss to root and show alert
            self.saveToPhotosThenAlert(outputURL: outputURL)
            
            self.assetWriter = nil
            self.assetWriterInput = nil
            self.pixelBufferAdaptor = nil
            self.recordingStartTime = nil
        }
    }
    
    /// Dismiss to root, then present an alert from the root controller
    private func dismissToRootAndPresentAlert(_ alert: UIAlertController) {
        DispatchQueue.main.async {
            // Capture a reference to the root after dismiss
            let presentAlert = {
                if let root = self.view.window?.rootViewController {
                    // Find the top-most from root
                    var top = root
                    while let presented = top.presentedViewController { top = presented }
                    top.present(alert, animated: true)
                } else {
                    // Fallback to keyWindow root if available
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
    
    // Save to Photos, then dismiss to main and show alert there
    private func saveToPhotosThenAlert(outputURL: URL) {
        Task { @MainActor in
            await withCheckedContinuation { continuation in
                self.ensurePhotoLibraryPermission { granted in
                    Task { @MainActor in
                        guard granted else {
                            // Dismiss to root then show denied alert
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
        captureSession.beginConfiguration()

        captureSession.sessionPreset = .inputPriority
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else { return }
        
        // Prefer well-known high-speed resolutions (1080p, then 720p) when multiple formats share the same max fps.
        let preferredDims: [(width: Int32, height: Int32)] = [
            (1920, 1080), (1080, 1920), // 1080p landscape/portrait
            (1280, 720), (720, 1280)    // 720p landscape/portrait
        ]
        
        func maxFPS(for format: AVCaptureDevice.Format) -> Double {
            return format.videoSupportedFrameRateRanges.map { $0.maxFrameRate }.max() ?? 0
        }
        
        // Find highest fps across all formats
        let globalMaxFPS = device.formats.map { maxFPS(for: $0) }.max() ?? 0
        
        // Filter formats that can reach the global max fps
        let highestFPSFormats = device.formats.filter { maxFPS(for: $0) == globalMaxFPS }
        
        // Among those, prefer preferredDims ordering
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
        
        // Choose final format: preferred if available, otherwise the format with the highest fps and largest resolution
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
            do {
                try device.lockForConfiguration()
                device.activeFormat = bestFormat
                // Remember best format dimensions for encoding
                let desc = CMVideoFormatDescriptionGetDimensions(bestFormat.formatDescription)
                self.activeDimensions = desc
                // Find the frame rate range with maxFrameRate and set min/max frame duration accordingly
                if let frameRateRange = bestFormat.videoSupportedFrameRateRanges.first(where: { $0.maxFrameRate == maxFrameRate }) {
                    let duration = CMTimeMake(value: 1, timescale: Int32(frameRateRange.maxFrameRate.rounded()))
                    device.activeVideoMinFrameDuration = duration
                    device.activeVideoMaxFrameDuration = duration
                }
                device.unlockForConfiguration()
                
                // Derive the actual active fps from the frame duration we set, for accuracy
                let actualDuration = device.activeVideoMaxFrameDuration
                if actualDuration.seconds > 0 {
                    self.activeFrameRate = 1.0 / actualDuration.seconds
                } else {
                    self.activeFrameRate = Double(maxFrameRate)
                }
                
                // Debug: print selected mode
                let dims = self.activeDimensions
                print("[Camera] Selected format: \(dims.width)x\(dims.height) @ ~\(self.activeFrameRate) fps (max \(maxFrameRate))")
            } catch {
                print("Failed to configure device for highest frame rate: \(error)")
            }
        }
        
        do {
            // Use .inputPriority so our chosen activeFormat + frame durations are honored over presets
            videoDeviceInput = try AVCaptureDeviceInput(device: device)
        } catch {
            print("Error creating device input: \(error)")
            return
        }
        
        guard videoDeviceInput != nil else { return }
        captureSession.addInput(videoDeviceInput)
        
        videoOutput = AVCaptureVideoDataOutput()
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        videoOutput.alwaysDiscardsLateVideoFrames = false
        
        if let connection = videoOutput.connection(with: .video) {
            if #available(iOS 17.0, *) {
                // Prefer modern rotation API
                let desiredAngle: CGFloat = 90 // portrait
                if connection.isVideoRotationAngleSupported(desiredAngle) {
                    connection.videoRotationAngle = desiredAngle
                }
            } else {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }
            
            // Live capture preview is using UIKit layer coordinates (top-left origin), so overlays should not flip Y.
            self.mirrored = (videoDeviceInput.device.position == .front)
            self.contentFlippedVertically = false
        }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            // Ensure we are on main thread for any UIView/Layer access
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
            
            self.ballTrailNormalized = results.compactMap { obs in
                if obs.labels.first?.identifier == "baseball" {
                    return CGPoint(x: obs.boundingBox.midX, y: obs.boundingBox.midY)
                }
                return nil
            }
            
            DispatchQueue.main.async {
                self.overlayLayer.sublayers?.forEach { $0.removeFromSuperlayer() }
                for obs in results {
                    if let confidence = obs.labels.first?.confidence,
                       self.allowedClasses.contains(obs.labels.first?.identifier ?? "") {
                        self.drawBoundingBox(obs.boundingBox, bufferSize: self.view.bounds.size,
                                             confidence: confidence, mirrored: self.mirrored,
                                             contentFlippedVertically: self.contentFlippedVertically)
                    }
                }
                self.drawBallTrail(bufferSize: self.view.bounds.size, mirrored: self.mirrored,
                                   contentFlippedVertically: self.contentFlippedVertically)
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
        let boxLayer = CAShapeLayer()
        boxLayer.frame = boxRect
        boxLayer.borderWidth = 2
        boxLayer.borderColor = UIColor.red.cgColor
        
        let textLayer = CATextLayer()
        textLayer.string = String(format: "%.2f", confidence)
        textLayer.foregroundColor = UIColor.red.cgColor
        textLayer.fontSize = 14
        textLayer.frame = CGRect(x: 0, y: -18, width: boxRect.width, height: 18)
        textLayer.contentsScale = view.traitCollection.displayScale
        boxLayer.addSublayer(textLayer)
        
        overlayLayer.addSublayer(boxLayer)
    }
    
    private func drawBallTrail(bufferSize: CGSize, mirrored: Bool,
                               contentFlippedVertically: Bool) {
        guard !ballTrailNormalized.isEmpty else { return }
        let trailLayer = CALayer()
        trailLayer.frame = CGRect(origin: .zero, size: bufferSize)
        
        for (i, p) in ballTrailNormalized.enumerated() {
            let point = convertPoint(p, bufferSize: bufferSize, mirrored: mirrored,
                                     contentFlippedVertically: contentFlippedVertically)
            let dot = CAShapeLayer()
            let radius: CGFloat = 5
            dot.path = UIBezierPath(ovalIn: CGRect(x: point.x - radius, y: point.y - radius,
                                                   width: radius * 2, height: radius * 2)).cgPath
            dot.fillColor = UIColor.red.withAlphaComponent(CGFloat(i) / CGFloat(maxTrailLength)).cgColor
            trailLayer.addSublayer(dot)
        }
        
        overlayLayer.addSublayer(trailLayer)
    }

    // MARK: - Export Annotated Video
    private func exportAnnotatedVideo(asset: AVAsset, track: AVAssetTrack) async {
        do {
            let reader = try AVAssetReader(asset: asset)
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
            
            // Preserve timing so the nominalFrameRate remains the high fps in the saved asset
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

            self.ballTrailNormalized.removeAll()

            while reader.status == .reading {
                guard let sampleBuffer = readerOutput.copyNextSampleBuffer(),
                      let px = CMSampleBufferGetImageBuffer(sampleBuffer) else { break }

                autoreleasepool {
                    let req = VNCoreMLRequest(model: self.visionModel)
                    let handler = VNImageRequestHandler(cvPixelBuffer: px, options: [:])
                    try? handler.perform([req])
                    let results = (req.results as? [VNRecognizedObjectObservation]) ?? []

                    for obs in results {
                        if obs.labels.first?.identifier == "baseball" {
                            let c = CGPoint(x: obs.boundingBox.midX, y: obs.boundingBox.midY)
                            self.ballTrailNormalized.append(c)
                            if self.ballTrailNormalized.count > self.maxTrailLength { self.ballTrailNormalized.removeFirst() }
                        }
                    }

                    let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                    while !writerInput.isReadyForMoreMediaData { usleep(1000) }
                    // Always compose annotation overlays on the pixel buffer before appending
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
    // Note: This method is still used by file export flow.
    // For live recording save flow, use saveToPhotosThenAlert(outputURL:)
    @MainActor
    private func saveAndShowExportCompletion(outputURL: URL) async {
        // Save to Photos with permission handling, then notify and show alert
        await withCheckedContinuation { continuation in
            self.ensurePhotoLibraryPermission { granted in
                
                Task { @MainActor in
                    guard granted else {
                        // User did NOT grant permissions: show alert and do NOT pass localIdentifier
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
                        // Request creation and capture localIdentifier for later use
                        if let placeholder = PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: outputURL)?.placeholderForCreatedAsset {
                            localIdentifier = placeholder.localIdentifier
                        }
                    }) { success, error in
                        Task { @MainActor in
                            if success, let id = localIdentifier {
                                // Successfully saved video to Photos
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
                                // Failed to save video to Photos: show error alert and notify with empty string
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

        // Create CIImage from source buffer
        let srcImage = CIImage(cvPixelBuffer: pixelBuffer)

        // Destination rect for drawing
        let rect = CGRect(x: 0, y: 0,
                          width: CVPixelBufferGetWidth(output),
                          height: CVPixelBufferGetHeight(output))

        // Draw overlays using Core Graphics
        UIGraphicsBeginImageContextWithOptions(rect.size, false, 1.0)
        guard let ctx = UIGraphicsGetCurrentContext() else {
            UIGraphicsEndImageContext()
            return output
        }

        // Draw the current frame without altering vertical orientation
        let uiImage = UIImage(ciImage: srcImage)
        uiImage.draw(in: rect)

        // Draw bounding boxes for allowed classes
        for obs in observations {
            guard let id = obs.labels.first?.identifier,
                  allowedClasses.contains(id) else { continue }
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
            ctx.setLineWidth(2)
            ctx.stroke(drawRect)
        }

        // Draw ball trail
        if !ballTrailNormalized.isEmpty {
            for (i, p) in ballTrailNormalized.enumerated() {
                var x = p.x * rect.width
                var y = p.y * rect.height
                if !contentFlippedVertically {
                    y = rect.height - y
                }
                
                if mirrored { x = rect.width - x }
                let radius: CGFloat = 5
                let alpha = CGFloat(i) / CGFloat(maxTrailLength)
                ctx.setFillColor(UIColor.red.withAlphaComponent(alpha).cgColor)
                ctx.fillEllipse(in: CGRect(x: x - radius, y: y - radius, width: radius * 2, height: radius * 2))
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
            // Notify callback with file path (legacy behavior)
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
                
                let angle = atan2(Double(t.b), Double(t.a)) // radians
                var degrees = Int(round(angle * 180.0 / .pi))
                // Normalize to [0, 360)
                degrees = (degrees % 360 + 360) % 360
                // Snap to nearest right angle (0, 90, 180, 270)
                let snapped: Int = {
                    let options = [0, 90, 180, 270]
                    let diffs = options.map { abs($0 - degrees) }
                    if let idx = diffs.enumerated().min(by: { $0.element < $1.element })?.offset {
                        return options[idx]
                    }
                    return 0
                }()

                // If a < 0 when rotation is 0/180, or c and b signs imply flip when 90/270.
                var mirrored = false
                switch snapped {
                case 0, 180:
                    mirrored = (t.a < 0)
                case 90, 270:
                    // For portrait-like rotations, horizontal flip manifests on the vertical axis component
                    mirrored = (t.d < 0)
                default:
                    mirrored = false
                }
                self.mirrored = mirrored

                // Upside down if rotation is 180
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
        guard videoURL == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform(self.requests)
        
        if isRecording {
            guard let assetWriter = assetWriter,
                  let assetWriterInput = assetWriterInput,
                  let pixelBufferAdaptor = pixelBufferAdaptor else { return }
            
            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            
            if recordingStartTime == nil {
                assetWriter.startWriting()
                assetWriter.startSession(atSourceTime: pts)
                recordingStartTime = pts
            }
            
            guard assetWriterInput.isReadyForMoreMediaData else { return }
            
            // Always compose annotation overlays on the pixel buffer
            // This ensures that the recorded video includes overlays such as bounding boxes and ball trails.
            // The pixel buffer is not rotated here; it matches the orientation of the live preview.
            let annotatedBuffer = makeAnnotatedPixelBuffer(from: pixelBuffer,
                                                           observations: lastObservations,
                                                           pool: pixelBufferAdaptor.pixelBufferPool)
            
            if let annotatedBuffer = annotatedBuffer {
                pixelBufferAdaptor.append(annotatedBuffer, withPresentationTime: pts)
            } else {
                pixelBufferAdaptor.append(pixelBuffer, withPresentationTime: pts)
            }
        }
    }
}

// MARK: - UIViewController extension for safe alert presentation
private extension UIViewController {
    /// Returns the top-most presented view controller starting from this view controller.
    func topMostViewController() -> UIViewController {
        var top = self
        while let presented = top.presentedViewController {
            top = presented
        }
        return top
    }
}

