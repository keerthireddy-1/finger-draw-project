# ✋ Finger Draw

An AI-powered air drawing application that lets you draw on screen using just your finger — no mouse, no stylus. Uses **MediaPipe** for real-time hand tracking and a **KNN classifier** to recognize hand-drawn digits.

---

## 🎥 Demo

> Draw digits in the air and watch them get recognized in real time!

---

## ✨ Features

- 🖊️ **Air Drawing** — Draw on a virtual canvas using your index finger via webcam
- 🎨 **Color Switching** — Change brush color using finger gestures (no buttons needed)
- 🔢 **Digit Recognition** — Press `D` to recognize any digit you've drawn using a KNN ML model
- 🧹 **Canvas Clearing** — Clear the canvas with a gesture or keyboard shortcut
- 📷 **Real-time Hand Tracking** — Powered by MediaPipe's hand landmark detection

---

## 🖐️ Gesture Controls

| Gesture | Action |
|---|---|
| ☝️ Index finger only | **Draw** on canvas |
| ✌️ 2 fingers up | Switch color to **Blue** |
| 🤟 3 fingers up | Switch color to **Green** |
| 🖖 4 fingers up | Switch color to **Red** |
| 🖐️ All 5 fingers | **Clear** the canvas |

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `D` | Detect & recognize drawn digit |
| `C` | Clear the canvas |
| `Q` | Quit the application |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Hand Tracking | [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) |
| Computer Vision | [OpenCV](https://opencv.org/) |
| Digit Recognition | [scikit-learn](https://scikit-learn.org/) — KNN Classifier |
| Dataset | `sklearn.datasets.load_digits` (8×8 pixel digit images) |
| Language | Python 3 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- A webcam

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/finger-draw.git
   cd finger-draw
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python mediapipe scikit-learn numpy
   ```

3. **Run the app**
   ```bash
   python finger_draw.py
   ```

---

## 📁 Project Structure

```
finger-draw/
│
├── finger_draw.py      # Main application script
└── README.md           # Project documentation
```

---

## 🧠 How It Works

1. **Hand Detection** — MediaPipe detects 21 hand landmarks from the webcam feed in real time.
2. **Gesture Recognition** — The number of raised fingers is computed by comparing fingertip and knuckle landmark positions.
3. **Drawing** — When only the index finger is raised, its tip coordinates are used to draw lines onto a persistent canvas overlay.
4. **Digit Recognition** — On pressing `D`, the canvas is grayscaled, thresholded, and resized to 8×8 pixels to match the `sklearn` digit dataset format. A pre-trained KNN model then predicts the digit.

---

## ⚠️ Known Limitations

- Digit recognition accuracy depends on drawing style — the model is trained on small 8×8 images, so writing neatly and centered improves results.
- Thumb gesture detection may behave differently for left-handed users (currently tuned for right hands).
- Performance may vary in poor lighting conditions.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for:

- Improved digit recognition (e.g., CNN model)
- Multi-hand support
- More gesture actions
- UI improvements
