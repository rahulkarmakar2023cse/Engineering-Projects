# scripts/encode_faces.py
import dlib
import cv2
import numpy as np
import os

def encode_faces(input_dir, face_rec_model_path, predictor_path):
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

    encodings = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            for rect in rects:
                shape = shape_predictor(gray, rect)
                face_encoding = face_rec_model.compute_face_descriptor(image, shape)
                encodings.append(np.array(face_encoding))

    return encodings

if __name__ == '__main__':
    input_dir = '../output/faces/bill_gates_gates'
    face_rec_model_path = '../models/dlib_face_recognition_resnet_model_v1.dat'
    predictor_path = '../models/shape_predictor_68_face_landmarks.dat'

    encodings = encode_faces(input_dir, face_rec_model_path, predictor_path)
    np.save('../output/encodings/bill_gates_gates_encodings.npy', encodings)
    print("Encodings saved to ../output/encodings/bill_gates_gates_encodings.npy")
