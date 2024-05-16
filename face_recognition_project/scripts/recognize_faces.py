# scripts/recognize_faces.py
import dlib
import cv2
import numpy as np
import time

def recognize_faces(video_path, encodings_paths, face_rec_model_path, predictor_path, frame_skip=5, resize_factor=0.5):
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
    known_encodings = {name: np.load(path) for name, path in encodings_paths.items()}

    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    detected_faces = {name: False for name in known_encodings.keys()}
    
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        frame_number += 1
        if not ret:
            break
        
        if frame_number % frame_skip != 0:
            continue
        
        # Resize frame
        frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        for rect in rects:
            shape = shape_predictor(gray, rect)
            face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape))
            
            for name, encoding in known_encodings.items():
                matches = np.linalg.norm(encoding - face_encoding, axis=1)
                if np.any(matches < 0.6):  # This threshold can be adjusted
                    (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detected_faces[name] = True
                    print(f"{name} detected in frame {frame_number}")
        
        cv2.imshow('Source Footage', frame)
        
        # Ensure the video runs at 24fps
        elapsed_time = time.time() - start_time
        wait_time = max(1, int((1.0 / 24 - elapsed_time) * 1000))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    for name, detected in detected_faces.items():
        if not detected:
            print(f"{name} not detected in the video")

if __name__ == '__main__':
    video_path = '../face_recognition_project/data/videos/sample.mp4'
    encodings_paths = {
        'bill_gates': '../face_recognition_project/output/encodings/bill_gates_encodings.npy',
        'sam_altman_altman': '../face_recognition_project/output/encodings/sam_altman_encodings.npy'
    }
    face_rec_model_path = '../face_recognition_project/models/dlib_face_recognition_resnet_model_v1.dat'
    predictor_path = '../face_recognition_project/models/shape_predictor_68_face_landmarks.dat'

    recognize_faces(video_path, encodings_paths, face_rec_model_path, predictor_path)