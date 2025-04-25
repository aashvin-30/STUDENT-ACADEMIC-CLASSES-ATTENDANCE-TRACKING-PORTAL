import os
#to delete a database file
# from datetime import datetime
DB_PATH = "instance/attendance.db"

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print("🗑️ Database deleted successfully.")
else:
    print("⚠️ Database file not found.")
