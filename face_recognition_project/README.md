# Face Recognition from Video

This project implements a face recognition system designed to identify specific individuals (e.g., Bill Gates and Sam Altman) from a video file. The system extracts faces from images, encodes them, and then uses these encodings to recognize the individuals in a video.

## Project Structure

```
face_recognition_project/
│
├── data/
│   ├── images/
│   │   ├── bill_gates/          # Directory containing images of Bill Gates
│   │   └── sam_altman/          # Directory containing images of Sam Altman
│   └── videos/
│       └── sample.mp4           # The video file to be processed
│
├── models/                      # Directory containing pre-trained models
│   ├── shape_predictor_68_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
│
├── output/
│   ├── encodings/               # Directory for storing face encodings
│   │   ├── bill_gates_encodings.npy
│   │   └── sam_altman_encodings.npy
│   └── faces/
│       ├── bill_gates/          # Directory for storing extracted faces of Bill Gates
│       └── sam_altman/          # Directory for storing extracted faces of Sam Altman
│
├── scripts/
│   ├── extract_faces.py         # Script to extract faces from images
│   ├── encode_faces.py          # Script to encode extracted faces
│   ├── recognize_faces.py       # Script to recognize faces in video
│   └── main.py                  # Main script to run the entire process
│
├── requirements.txt             # List of dependencies
└── README.md                    # This README file
```

## Requirements

- Python 3.6+
- dlib
- OpenCV
- NumPy

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Setup

1. **Collect Images of Individuals**:
   - Place multiple images of Bill Gates in the `data/images/bill_gates/` directory.
   - Place multiple images of Sam Altman in the `data/images/sam_altman/` directory.

2. **Place the Video**:
   - Place the video file (`sample.mp4`) in the `data/videos/` directory.

3. **Pre-trained Models**:
   - Download the following pre-trained models and place them in the `models/` directory:
     - [`shape_predictor_68_face_landmarks.dat`](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
     - [`dlib_face_recognition_resnet_model_v1.dat`](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)

## How to Run

Run the main script to perform face extraction, encoding, and recognition:

```bash
python scripts/main.py
```

## Algorithms Used

### 1. Face Detection and Extraction

- **Algorithm**: Histogram of Oriented Gradients (HOG) and Linear SVM
- **Library**: dlib
- **Details**: The `extract_faces.py` script uses dlib's HOG-based face detector to detect faces in the images. Detected faces are then extracted and saved as separate images.

```python
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
rects = detector(gray, 1)
```

### 2. Face Encoding

- **Algorithm**: Deep Learning-based Face Recognition (ResNet)
- **Library**: dlib
- **Details**: The `encode_faces.py` script uses dlib's ResNet-based face recognition model to generate 128-dimensional face embeddings (encodings) for each extracted face.

```python
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
face_encoding = face_rec_model.compute_face_descriptor(image, shape)
```

### 3. Face Recognition in Video

- **Algorithm**: Euclidean Distance
- **Library**: OpenCV, dlib
- **Details**: The `recognize_faces.py` script captures frames from the video, detects faces, computes their encodings, and compares them to the known encodings using Euclidean distance. If the distance is below a certain threshold, the face is recognized.

```python
matches = np.linalg.norm(encoding - face_encoding, axis=1)
if np.any(matches < 0.6):  # This threshold can be adjusted
    (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

### Optimizations for Video Processing

- **Frame Skipping**: Only process every nth frame to reduce computational load.
- **Frame Resizing**: Resize frames to a smaller size before processing to speed up detection and recognition.

```python
frame_skip = 5
resize_factor = 0.5
if frame_number % frame_skip != 0:
    continue
frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
```

### Running the Project at 24fps

To ensure the video runs at 24 frames per second, the script uses `cv2.waitKey()` with a calculated wait time based on the processing time of each frame.

```python
start_time = time.time()
# Process frame
elapsed_time = time.time() - start_time
wait_time = max(1, int((1.0 / 24 - elapsed_time) * 1000))
if cv2.waitKey(wait_time) & 0xFF == ord('q'):
    break
```
