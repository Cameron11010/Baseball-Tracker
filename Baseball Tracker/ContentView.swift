import SwiftUI
import PhotosUI
import AVFoundation
import AVKit
import Photos

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var showingCamera = false
    @State private var lastVideoIdentifier: String?
    @State private var lastVideoThumbnail: UIImage?
    @State private var showingLastVideo = false
    @State private var currentVideoURL: URL? // Video to send to CameraView
    @State private var isRecordingLive = false // Track live recording
    @State private var pickedVideo: PickedVideo?

    var body: some View {
        GeometryReader { geo in
            ZStack {
                VStack(spacing: geo.size.height * 0.03) {
                    
                    
                    if let identifier = lastVideoIdentifier, !identifier.isEmpty {
                        HStack {
                            Spacer()
                            PhotosVideoPlayer(localIdentifier: identifier, showControls: false, loop: true)
                                .frame(width: geo.size.width * 0.3,
                                       height: geo.size.width * 0.3 * 9/16)
                                .cornerRadius(12)
                                .shadow(radius: 3)
                                .onTapGesture { showingLastVideo = true }
                        }
                        .padding(.top, geo.safeAreaInsets.top + 10)
                        .padding(.trailing, 16)
                    } else if let thumbnail = lastVideoThumbnail {
                        // fallback thumbnail image
                        HStack {
                            Spacer()
                            Image(uiImage: thumbnail)
                                .resizable()
                                .scaledToFill()
                                .frame(width: geo.size.width * 0.3,
                                       height: geo.size.width * 0.3 * 9/16)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .overlay(RoundedRectangle(cornerRadius: 12)
                                    .stroke(Color.white, lineWidth: 2))
                                .shadow(radius: 3)
                        }
                        .padding(.top, geo.safeAreaInsets.top + 10)
                        .padding(.trailing, 16)
                    }

                    Spacer(minLength: geo.size.height * 0.1)

                    // Open live camera with overlay for detecting baseballs
                    Button("Open Camera") {
                        currentVideoURL = nil
                        showingCamera = true
                    }
                    .font(.headline)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(geo.size.height * 0.03)

                    // Select from library
                    PhotosPicker(selection: $selectedItem,
                                 matching: .videos,
                                 photoLibrary: .shared()) {
                        Text("Select Video")
                            .font(.headline)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(geo.size.height * 0.03)
                    }

                    Spacer()
                }
                .padding()
            }
        }
        .fullScreenCover(isPresented: $showingCamera) {
            CameraView(
                onVideoSaved: { identifier in
                    Task { @MainActor in
                        lastVideoIdentifier = identifier
                        showingCamera = false
                        print("✅ Video saved. Returning to home with identifier: \(identifier)")
                    }
                },
                videoURL: currentVideoURL,
                isRecordingLive: $isRecordingLive
            )
            .ignoresSafeArea()
        }
        .fullScreenCover(item: $pickedVideo) {
            CameraView(
                onVideoSaved: nil,
                videoURL: $0.url,
                isRecordingLive: .constant(false)
            )
            .ignoresSafeArea()
        }
        .fullScreenCover(isPresented: $showingLastVideo) {
            if let identifier = lastVideoIdentifier, !identifier.isEmpty {
                PhotosVideoPlayer(localIdentifier: identifier, showControls: true, loop: false)
                    .ignoresSafeArea()
            }
        }
        .onChange(of: selectedItem) { oldItem, newItem in
            print("onChange(selectedItem): \(String(describing: newItem))")
            Task { await loadSelectedVideo(newItem) }
        }
        .onAppear {
            print("ContentView appeared")
        }
    }

    // MARK: - Load selected video from PhotosPicker
    private func loadSelectedVideo(_ item: PhotosPickerItem?) async {
        print("loadSelectedVideo: selection changed -> \(String(describing: item))")
        guard let item = item else { return }

        func copyToStableTemp(from sourceURL: URL) throws -> URL {
            let ext = sourceURL.pathExtension.isEmpty ? "mp4" : sourceURL.pathExtension
            let destURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(ext)
            if FileManager.default.fileExists(atPath: destURL.path) {
                try FileManager.default.removeItem(at: destURL)
            }
            try FileManager.default.copyItem(at: sourceURL, to: destURL)
            return destURL
        }

        do {
            var finalURL: URL?

            if let url = try await item.loadTransferable(type: URL.self) {
                print("loadSelectedVideo: got URL via Transferable -> \(url)")
                do {
                    finalURL = try copyToStableTemp(from: url)
                } catch {
                    print("Copy to temp failed (Transferable URL): \(error). Using original URL.")
                    finalURL = url
                }
            }

            if finalURL == nil {
                if let data = try await item.loadTransferable(type: Data.self) {
                    let ext = item.supportedContentTypes.first?.preferredFilenameExtension ?? "mov"
                    let tempURL = FileManager.default.temporaryDirectory
                        .appendingPathComponent(UUID().uuidString)
                        .appendingPathExtension(ext)
                    try data.write(to: tempURL, options: .atomic)
                    finalURL = tempURL
                    print("loadSelectedVideo: wrote Data fallback to temp -> \(tempURL)")
                }
            }

            guard let resolvedURL = finalURL else {
                print("loadSelectedVideo: failed to resolve a URL for picked video")
                return
            }

            await MainActor.run {
                currentVideoURL = resolvedURL
                pickedVideo = PickedVideo(url: resolvedURL)
                selectedItem = nil
                lastVideoIdentifier = nil
                print("loadSelectedVideo: presenting CameraView directly with file URL -> \(resolvedURL)")
            }

        } catch {
            print("Error loading selected video: \(error)")
        }
    }

    // MARK: - Generate thumbnail (fallback only)
    private func generateThumbnail(url: URL) async -> UIImage? {
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 300, height: 300)
        let time = CMTime(seconds: 0.1, preferredTimescale: 600)

        if #available(iOS 18.0, *) {
            return await withCheckedContinuation { continuation in
                generator.generateCGImageAsynchronously(for: time) { cgImage, _, _ in
                    if let cgImage = cgImage {
                        continuation.resume(returning: UIImage(cgImage: cgImage))
                    } else {
                        continuation.resume(returning: nil)
                    }
                }
            }
        } else {
            return await withCheckedContinuation { continuation in
                generator.generateCGImagesAsynchronously(forTimes: [NSValue(time: time)]) { _, cgImage, _, _, _ in
                    if let cgImage = cgImage {
                        continuation.resume(returning: UIImage(cgImage: cgImage))
                    } else {
                        continuation.resume(returning: nil)
                    }
                }
            }
        }
    }
}

struct PickedVideo: Identifiable {
    let id = UUID()
    let url: URL
}

struct PhotosVideoPlayer: View {
    let localIdentifier: String
    var showControls: Bool = false
    var loop: Bool = false
    @State private var player: AVPlayer?

    var body: some View {
        Group {
            if let player = player {
                VideoPlayer(player: player)
                    .onAppear { player.play() }
                    .onDisappear { player.pause() }
            } else {
                ProgressView()
                    .task { await fetchPlayerWithRetry() }
            }
        }
        .animation(.easeIn(duration: 0.3), value: player != nil)
    }

    private func fetchPlayerWithRetry() async {
        for attempt in 1...6 { // retry up to ~3s total
            if await fetchPlayer() {
                print("✅ Player loaded after \(attempt) attempt(s)")
                return
            } else {
                print("⏳ Retrying player fetch (\(attempt))...")
                try? await Task.sleep(nanoseconds: 500_000_000) // 0.5s
            }
        }
        print("⚠️ Failed to load player after retries")
    }

    @MainActor
    private func fetchPlayer() async -> Bool {
        let assets = PHAsset.fetchAssets(withLocalIdentifiers: [localIdentifier], options: nil)
        guard let asset = assets.firstObject else { return false }

        let options = PHVideoRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = true

        // Bridge the callback API to async/await
        return await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            PHImageManager.default().requestPlayerItem(forVideo: asset, options: options) { playerItem, _ in
                if let item = playerItem {
                    let avPlayer = AVPlayer(playerItem: item)
                    if loop {
                        NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime,
                                                               object: item,
                                                               queue: .main) { _ in
                            avPlayer.seek(to: .zero)
                            avPlayer.play()
                        }
                    }
                    DispatchQueue.main.async {
                        self.player = avPlayer
                        avPlayer.play()
                        continuation.resume(returning: true)
                    }
                } else {
                    continuation.resume(returning: false)
                }
            }
        }
    }
}
