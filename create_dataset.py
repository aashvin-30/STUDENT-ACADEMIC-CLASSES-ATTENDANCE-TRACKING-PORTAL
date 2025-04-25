import cv2
import os
import mediapipe as mp
import numpy as np
from dataset_utils import save_user_details, apply_clahe

# Constants
DATASET_DIR = "dataSet"
os.makedirs(DATASET_DIR, exist_ok=True)

# Initialize MediaPipe face detector
mp_face_detection = mp.solutions.face_detection
#increme limt 100 to 400
def create_dataset(user_id, user_name, num_samples=100):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    save_user_details(str(user_id), user_name)

    # Delete existing images for the user
    for file in os.listdir(DATASET_DIR):
        if file.startswith(f"User.{user_id}."):
            os.remove(os.path.join(DATASET_DIR, file))

    sample_num = 0
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        while sample_num < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bbox.xmin * iw)
                    y = int(bbox.ymin * ih)
                    w = int(bbox.width * iw)
                    h = int(bbox.height * ih)

                    # Add margin
                    margin = 30
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(iw, x + w + margin)
                    y2 = min(ih, y + h + margin)

                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue

                    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    face_gray = cv2.resize(face_gray, (200, 200))
                    face_gray = apply_clahe(face_gray)

                    sample_num += 1
                    filename = f"{DATASET_DIR}/User.{user_id}.{sample_num}.jpg"
                    cv2.imwrite(filename, face_gray)

                    cv2.putText(frame, f"Sample: {sample_num}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Capturing Dataset", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Dataset for {user_name} (ID: {user_id}) collected successfully!")
    print(f"📸 {sample_num} samples saved in {DATASET_DIR}.")