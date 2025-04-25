from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from pytz import timezone

db = SQLAlchemy()

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), nullable=False)
    student_name = db.Column(db.String(100), nullable=False)
    engagement = db.Column(db.String(50), default="Present")
    location = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone("Asia/Kolkata")))
