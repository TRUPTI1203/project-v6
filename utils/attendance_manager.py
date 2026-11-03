import os
import csv
import threading
from datetime import datetime


class AttendanceManager:
    """Persist attendance and prevent duplicate entries for the same day."""

    def __init__(self):
        self._lock = threading.Lock()
        self.date = datetime.now().strftime("%Y-%m-%d")

        self.folder = os.path.join("data", "attendance")
        os.makedirs(self.folder, exist_ok=True)

        self.file = os.path.join(self.folder, f"{self.date}.csv")
        self.marked_today = set()

        self._ensure_file()
        self._load_marked_today()

    def _ensure_file(self):
        if os.path.exists(self.file) and os.path.getsize(self.file) > 0:
            return

        with open(self.file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Roll",
                "Name",
                "Branch",
                "Semester",
                "Section",
                "Date",
                "Time",
                "Status",
            ])

    def _load_marked_today(self):
        """Load today's already-present roll numbers from the CSV.

        The previous version kept this only in memory. Every time the attendance
        camera was started, a new AttendanceManager was created and the set became
        empty, allowing the same student to be appended again. Loading the CSV
        makes duplicate prevention survive camera restarts and page navigation.
        """
        try:
            with open(self.file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    roll = str(row.get("Roll", "")).strip()
                    status = str(row.get("Status", "")).strip().lower()
                    if roll and status == "present":
                        self.marked_today.add(roll)
        except (OSError, csv.Error):
            # If the file cannot be read, leave the in-memory set empty and let
            # mark() handle the write error normally.
            pass

    def mark(self, info):
        """Mark one student once for the current date.

        Returns True only when a new row was written. Returns False when that
        roll number is already present in today's attendance CSV.
        """
        roll = str(info["roll"]).strip()

        if not roll:
            return False

        with self._lock:
            # Fast in-memory check.
            if roll in self.marked_today:
                return False

            # Re-check the CSV immediately before writing. This protects against
            # another AttendanceManager instance having written the same student.
            try:
                with open(self.file, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_roll = str(row.get("Roll", "")).strip()
                        status = str(row.get("Status", "")).strip().lower()
                        if existing_roll == roll and status == "present":
                            self.marked_today.add(roll)
                            return False
            except (OSError, csv.Error):
                pass

            now = datetime.now()

            try:
                with open(self.file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        roll,
                        info["name"],
                        info["branch"],
                        info["semester"],
                        info["section"],
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S"),
                        "Present",
                    ])
            except OSError:
                return False

            self.marked_today.add(roll)
            return True
