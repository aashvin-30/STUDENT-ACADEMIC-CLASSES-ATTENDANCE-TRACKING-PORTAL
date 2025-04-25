import cv2
import os
import numpy as np
import json

DATASET_DIR = "dataSet"
RECOGNIZER_FILE = "recognizer.yml"
LABEL_MAP_FILE = "label_map.json"

def train_and_save_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_samples, numeric_ids = [], []
    label_to_id = {}
    id_to_label = {}
    current_id = 0

    for filename in sorted(os.listdir(DATASET_DIR)):
        if filename.endswith(".jpg") and "_color" not in filename:
            path = os.path.join(DATASET_DIR, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Skipping unreadable image: {filename}")
                continue
            try:
                img = cv2.resize(img, (200, 200))
                label = filename.split(".")[1]  # e.g., "78945CB123"

                if label not in label_to_id:
                    label_to_id[label] = current_id
                    id_to_label[current_id] = label
                    current_id += 1

                numeric_id = label_to_id[label]
                face_samples.append(img)
                numeric_ids.append(numeric_id)
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                continue

    if not face_samples:
        print("⚠️ No training data found.")
        return

    recognizer.train(face_samples, np.array(numeric_ids))
    recognizer.save(RECOGNIZER_FILE)

    # Save mapping
    with open(LABEL_MAP_FILE, "w") as f:
        json.dump(id_to_label, f, indent=4)

    print(f"✅ Trained {len(id_to_label)} users. Saved recognizer as '{RECOGNIZER_FILE}' and label map as '{LABEL_MAP_FILE}'")