# Baseball Tracker

An iOS app that detects a baseball in real time from the phone's camera feed, using a custom-trained YOLOv8 model converted to CoreML. This repo holds the working app and the Python side used to train the detector; a related repo, [Dissertation](https://github.com/Cameron11010/Dissertation), documents the full two-phone stereo-vision pitch-calling system this app is built towards.

## Structure

```
Baseball Tracker.xcodeproj/    Xcode project
Baseball Tracker/              App source (SwiftUI)
  Baseball_TrackerApp.swift
  ContentView.swift
  CameraView.swift             Camera capture + live detection overlay
  VideoPlayerView.swift
  best.mlpackage                Trained CoreML detector
  gradual updates/              Screen recordings of the detector improving over training iterations
Python training code/
  training.py                  YOLOv8 training script (Ultralytics)
  export.py                    Exports the trained model to CoreML
  baseball.yaml                Dataset config for training
  results/                     Training curves, confusion matrix, sample predictions
```

## How it works

- **Detection.** `training.py` trains a YOLOv8 model from scratch on a labelled baseball image set, then `export.py` converts the result to a `.mlpackage` for on-device inference via CoreML/Vision.
- **App.** `CameraView.swift` streams the camera feed and runs the exported model live, drawing detection boxes over the video as the ball moves through frame.

## Training data

The training set is the baseball collection from [images.cv](https://images.cv/download/baseball/1558), referenced in `Python training code/READ ME.txt`.

## Setup

**Training (optional — a trained model is already committed as `best.mlpackage`):**
```bash
cd "Python training code"
pip install ultralytics torch
# point baseball.yaml at your own dataset path, then:
python training.py
python export.py
```

**App:**
1. Open `Baseball Tracker.xcodeproj` in Xcode.
2. Build and run on a physical device — CoreML detection needs the camera and runs meaningfully faster on-device than in the simulator.

## Related

Part of the same project as [Dissertation](https://github.com/Cameron11010/Dissertation) (full stereo two-phone system) and [baseball-image-classifier](https://github.com/Cameron11010/baseball-image-classifier) (an earlier detection pass using stock COCO classes).

---

Built by [Cameron Millar](https://github.com/Cameron11010).
