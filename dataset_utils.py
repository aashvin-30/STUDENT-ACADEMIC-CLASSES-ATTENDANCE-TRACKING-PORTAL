import os
import json
import cv2

NAMES_FILE = "names.json"

def save_user_details(user_id, user_name):
    """Save user ID and name to names.json file"""
    data = {}

    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}

    data[str(user_id)] = user_name

    with open(NAMES_FILE, "w") as file:
        json.dump(data, file, indent=4)

def apply_clahe(gray_img):
    """Apply CLAHE to improve contrast in grayscale image"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)
