import cv2
import os
import json
import pandas as pd
import smtplib
from datetime import datetime
from models import Attendance, db
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import geocoder
from dotenv import load_dotenv
from pytz import timezone
import numpy as np

load_dotenv()

# === Config ===
RECOGNIZER_FILE = "recognizer.yml"
LABEL_MAP_FILE = "label_map.json"
NAMES_FILE = "names.json"
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "aashvins212@gmail.com")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = os.getenv("EMAIL_USER")
SMTP_PASSWORD = os.getenv("EMAIL_PASS")

india_tz = timezone("Asia/Kolkata")

# === Face Recognition Setup ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists(RECOGNIZER_FILE) and os.path.getsize(RECOGNIZER_FILE) > 0:
    try:
        recognizer.read(RECOGNIZER_FILE)
        print("✅ Recognizer model loaded.")
    except cv2.error as e:
        print(f"❌ Failed to load recognizer: {e}")
else:
    print("⚠️ Recognizer file missing or empty.")

# === Helpers ===

def apply_clahe(img):
    clahe = cv2.createCLAHE(2.0, (8, 8))
    return clahe.apply(img)

def get_location():
    g = geocoder.ip("me")
    return f"{g.city}, {g.country}" if g.city else "Unknown"

def mark_attendance(student_id, student_name):
    now = datetime.now(india_tz)
    today = now.date()

    existing = Attendance.query.filter(
        Attendance.student_id == student_id,
        Attendance.timestamp >= datetime(today.year, today.month, today.day, 0, 0, 0)
    ).first()

    if existing:
        return False

    entry = Attendance(
        student_id=student_id,
        student_name=student_name,
        engagement="Present",
        location=str("Bannari Amman Institute of Technology"),
        #location=get_location(),
        timestamp=now
    )
    db.session.add(entry)
    db.session.commit()
    print(f"✅ Attendance marked for {student_name} (ID: {student_id})")
    return True

def recognize_face():
    # Load ID to label map
    with open(LABEL_MAP_FILE) as f:
        id_to_label = json.load(f)

    with open(NAMES_FILE) as f:
        name_data = json.load(f)

    cap = cv2.VideoCapture(0)
    recognized = set()
    already_marked = set()
    last_recognized_id = None
    cooldown_counter = 0

    print("🧠 Face recognition started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))
            face = apply_clahe(face)

            try:
                numeric_id, conf = recognizer.predict(face)
                student_id = id_to_label.get(str(numeric_id), "Unknown")
                name = name_data.get(student_id, "Unknown")

                
                if conf < 55:
                    if student_id == last_recognized_id and cooldown_counter < 10:
                        continue

                    if student_id not in recognized:
                        if not mark_attendance(student_id, name):
                            print(f"⚠️ Already marked: {name}")
                            already_marked.add(name)
                        else:
                            print(f"✅ Marked attendance for {name} (Confidence: {conf:.2f})")
                        recognized.add(student_id)
                        last_recognized_id = student_id
                        cooldown_counter = 0

            except Exception as e:
                print(f"❌ Recognition error: {e}")
                continue

        cooldown_counter += 1
        cv2.imshow("Recognizing", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    send_email()
    return already_marked

def send_email():
    try:
        records = Attendance.query.order_by(Attendance.timestamp.desc()).all()
        df = pd.DataFrame([{
            "ID": r.student_id,
            "Name": r.student_name,
            "Time": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Location": r.location,
            "Engagement": r.engagement
        } for r in records])

        df.to_csv("temp_attendance.csv", index=False)

        msg = MIMEMultipart()
        msg["From"] = SMTP_USERNAME
        msg["To"] = ADMIN_EMAIL
        msg["Subject"] = "📊 Daily Attendance Report"

        with open("temp_attendance.csv", "rb") as file:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename="attendance.csv")
            msg.attach(part)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SMTP_USERNAME, ADMIN_EMAIL, msg.as_string())
        server.quit()

        os.remove("temp_attendance.csv")
        print("✅ Email sent successfully!")

    except Exception as e:
        print(f"❌ Email send failed: {e}")