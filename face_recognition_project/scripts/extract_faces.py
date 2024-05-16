# scripts/extract_faces.py
import dlib
import cv2
import os

def extract_faces(input_dir, output_dir, detector, predictor):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            for i, rect in enumerate(rects):
                shape = predictor(gray, rect)
                (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
                face = image[y:y + h, x:x + w]
                face_filename = f"{os.path.splitext(filename)[0]}_face_{i}.jpg"
                face_path = os.path.join(output_dir, face_filename)
                cv2.imwrite(face_path, face)
                print(f"Face {i} from {filename} saved to {face_path}")

if __name__ == '__main__':
    input_dir = '../data/images/bill_gates_gates'
    output_dir = '../output/faces/bill_gates_gates'
    os.makedirs(output_dir, exist_ok=True)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')

    extract_faces(input_dir, output_dir, detector, predictor)
