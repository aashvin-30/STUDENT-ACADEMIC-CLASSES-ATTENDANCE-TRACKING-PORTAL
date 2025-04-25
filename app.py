from flask import Flask, render_template, request, redirect, url_for, flash, make_response
import os, json, pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from train import train_and_save_model
from create_dataset import create_dataset
from recognizer_module import recognize_face, send_email
from models import db, Attendance
from apscheduler.schedulers.background import BackgroundScheduler
from io import BytesIO
from xhtml2pdf import pisa

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "supersecretkey")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///attendance.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
with app.app_context():
    db.create_all()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user_id = request.form["user_id"]
        user_name = request.form["user_name"]
        if not user_id or not user_name:
            flash("Please fill in all fields", "danger")
            return redirect(url_for("register"))

        try:
            create_dataset(user_id, user_name)
            train_and_save_model()
            flash(f"✅ Registered and trained: {user_name}", "success")
        except Exception as e:
            flash(f"❌ Error: {e}", "danger")
        return redirect(url_for("index"))
    return render_template("register.html")

@app.route("/detect", methods=["POST"])
def detect():
    already_marked = recognize_face()
    if already_marked:
        flash(f"⚠️ Already marked: {', '.join(already_marked)}", "warning")
    else:
        flash("✅ Attendance captured & emailed.", "success")
    return redirect(url_for("index"))

@app.route("/attendance")
def view_attendance():
    records = Attendance.query.order_by(Attendance.timestamp.desc()).all()
    df = pd.DataFrame([{
        "Student ID": r.student_id,
        "Name": r.student_name,
        "Engagement": r.engagement,
        "Location": r.location,
        "Timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    } for r in records])
    table = df.to_html(classes="table table-striped", index=False)
    return render_template("attendance.html", table=table)

@app.route("/dashboard")
def dashboard():
    with open("names.json") as f:
        name_data = json.load(f)
    records = Attendance.query.all()
    attendance_count = {}
    for record in records:
        name = record.student_name
        attendance_count[name] = attendance_count.get(name, 0) + 1
    return render_template("dashboard.html", students=name_data, attendance=attendance_count)

@app.route("/export/pdf")
def export_pdf():
    records = Attendance.query.order_by(Attendance.timestamp.desc()).all()
    df = pd.DataFrame([{
        "Student ID": r.student_id,
        "Name": r.student_name,
        "Engagement": r.engagement,
        "Location": r.location,
        "Timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    } for r in records])

    html_table = df.to_html(classes="table table-bordered", index=False)
    rendered_html = render_template("pdf_template.html", table=html_table)
    pdf = BytesIO()
    pisa.CreatePDF(BytesIO(rendered_html.encode("utf-8")), dest=pdf)

    response = make_response(pdf.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=attendance_report.pdf"
    return response

# Scheduler
def schedule_email():
    scheduler = BackgroundScheduler()
    scheduler.add_job(send_email, "cron", hour=18, minute=0)
    scheduler.start()

if __name__ == "__main__":
    schedule_email()
    app.run(debug=True)
