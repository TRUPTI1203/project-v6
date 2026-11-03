import os
import csv
from datetime import datetime
import customtkinter as ctk
import config


class HomePage(ctk.CTkFrame):
    """Dashboard with live student and attendance statistics."""

    def __init__(self, parent):
        super().__init__(parent, fg_color=config.BACKGROUND_COLOR)
        self._refresh_job = None
        self.create_widgets()
        self.refresh_stats()

    def create_widgets(self):
        title = ctk.CTkLabel(
            self,
            text="Welcome Administrator 👋",
            font=config.TITLE_FONT,
            text_color=config.TEXT_COLOR,
        )
        title.pack(anchor="w", padx=25, pady=(20, 5))

        subtitle = ctk.CTkLabel(
            self,
            text="Face Recognition Attendance Management Dashboard",
            font=config.SUBTITLE_FONT,
            text_color="gray40",
        )
        subtitle.pack(anchor="w", padx=28)

        cards = ctk.CTkFrame(self, fg_color="transparent")
        cards.pack(fill="x", padx=20, pady=30)

        self.registered_value = self.create_card(cards, "👥 Registered Students", "0")
        self.attendance_value = self.create_card(cards, "✅ Today's Attendance", "0")
        self.create_card(cards, "🤖 AI Model", "FaceNet")
        self.date_value = self.create_card(
            cards, "📅 Date", datetime.now().strftime("%d %b %Y")
        )

        info = ctk.CTkFrame(self, fg_color="white", corner_radius=12)
        info.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        ctk.CTkLabel(
            info, text="Project Overview", font=config.HEADING_FONT
        ).pack(anchor="w", padx=20, pady=(20, 10))

        features = [
            "✔ Automatic Face Recognition Attendance",
            "✔ Deep Learning using FaceNet",
            "✔ Computer Vision with OpenCV",
            "✔ Real-Time Student Recognition",
            "✔ Attendance Report Generation",
            "✔ Modern GUI using CustomTkinter",
        ]

        for item in features:
            ctk.CTkLabel(
                info, text=item, font=config.NORMAL_FONT, anchor="w"
            ).pack(anchor="w", padx=30, pady=4)

        self.status_label = ctk.CTkLabel(
            info,
            text="🟢 System Ready",
            font=config.BUTTON_FONT,
            text_color=config.SUCCESS_COLOR,
        )
        self.status_label.pack(anchor="w", padx=20, pady=20)

    def create_card(self, parent, title, value):
        card = ctk.CTkFrame(
            parent,
            width=180,
            height=120,
            corner_radius=15,
            fg_color="white",
        )
        card.pack(side="left", padx=10, fill="both", expand=True)
        card.pack_propagate(False)

        ctk.CTkLabel(
            card, text=title, font=("Segoe UI", 14, "bold")
        ).pack(pady=(20, 10))

        value_lbl = ctk.CTkLabel(
            card,
            text=value,
            font=("Segoe UI", 22, "bold"),
            text_color=config.PRIMARY_COLOR,
        )
        value_lbl.pack()
        return value_lbl

    @staticmethod
    def _registered_students_count():
        students_dir = os.path.join("data", "students")
        if not os.path.isdir(students_dir):
            return 0
        return sum(
            1
            for name in os.listdir(students_dir)
            if os.path.isdir(os.path.join(students_dir, name))
        )

    @staticmethod
    def _today_attendance_count():
        today = datetime.now().strftime("%Y-%m-%d")
        file = os.path.join("data", "attendance", f"{today}.csv")
        if not os.path.exists(file):
            return 0

        rolls = set()
        try:
            with open(file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Roll"):
                        rolls.add(str(row["Roll"]).strip())
        except (OSError, csv.Error):
            return 0
        return len(rolls)

    def refresh_stats(self):
        if not self.winfo_exists():
            return

        self.registered_value.configure(
            text=str(self._registered_students_count())
        )
        self.attendance_value.configure(
            text=str(self._today_attendance_count())
        )
        self.date_value.configure(text=datetime.now().strftime("%d %b %Y"))

        self._refresh_job = self.after(2000, self.refresh_stats)

    def destroy(self):
        if self._refresh_job is not None:
            try:
                self.after_cancel(self._refresh_job)
            except Exception:
                pass
            self._refresh_job = None
        super().destroy()
