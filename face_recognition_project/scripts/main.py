# scripts/main.py
import os
import numpy as np
import dlib
from extract_faces import extract_faces
from encode_faces import encode_faces
from recognize_faces import recognize_faces

def main():
    # Paths
    images_dirs = {
        'bill_gates': '../face_recognition_project/data/images/bill_gates',
        'sam_altman': '../face_recognition_project/data/images/sam_altman'
    }
    faces_output_dirs = {
        'bill_gates': '../face_recognition_project/output/faces/bill_gates',
        'sam_altman': '../face_recognition_project/output/faces/sam_altman'
    }
    encodings_paths = {
        'bill_gates': '../face_recognition_project/output/encodings/bill_gates_encodings.npy',
        'sam_altman': '../face_recognition_project/output/encodings/sam_altman_encodings.npy'
    }
    video_path = '../face_recognition_project/data/videos/sample.mp4'
    face_rec_model_path = '../face_recognition_project/models/dlib_face_recognition_resnet_model_v1.dat'
    predictor_path = '../face_recognition_project/models/shape_predictor_68_face_landmarks.dat'
    
    # Ensure output directories exist
    for dir in faces_output_dirs.values():
        os.makedirs(dir, exist_ok=True)
    os.makedirs('../face_recognition_project/output/encodings', exist_ok=True)
    
    # Step 1: Extract Faces from images
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    for name, dir in images_dirs.items():
        print(f"Extracting faces from {name}'s images...")
        extract_faces(dir, faces_output_dirs[name], detector, predictor)
    
    # Step 2: Encode Faces
    for name, dir in faces_output_dirs.items():
        print(f"Encoding faces of {name}...")
        encodings = encode_faces(dir, face_rec_model_path, predictor_path)
        np.save(encodings_paths[name], encodings)
    
    # Step 3: Recognize Faces in Video
    print("Recognizing faces in video...")
    recognize_faces(video_path, encodings_paths, face_rec_model_path, predictor_path)

if __name__ == '__main__':
    main()