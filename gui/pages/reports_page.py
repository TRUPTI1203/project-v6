import os
from datetime import datetime
import pandas as pd
import customtkinter as ctk
from tkinter import ttk

import config


COLUMNS = (
    "Roll",
    "Name",
    "Branch",
    "Semester",
    "Section",
    "Date",
    "Time",
    "Status",
)


class ReportsPage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color=config.BACKGROUND_COLOR)
        self.df = pd.DataFrame(columns=COLUMNS)
        self.create_widgets()
        self.load_attendance()

    def create_widgets(self):
        title = ctk.CTkLabel(
            self, text="📊 Attendance Reports", font=config.TITLE_FONT
        )
        title.pack(pady=(20, 5))

        subtitle = ctk.CTkLabel(
            self,
            text="View and search today's attendance records",
            font=config.SUBTITLE_FONT,
        )
        subtitle.pack(pady=(0, 15))

        search_frame = ctk.CTkFrame(self, fg_color="transparent")
        search_frame.pack(fill="x", padx=30)

        self.search_entry = ctk.CTkEntry(
            search_frame,
            width=300,
            placeholder_text="Search by Roll or Name",
        )
        self.search_entry.pack(side="left", padx=10)

        ctk.CTkButton(
            search_frame,
            text="🔍 Search",
            command=self.search_attendance,
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            search_frame,
            text="🔄 Refresh",
            command=self.load_attendance,
        ).pack(side="left", padx=5)

        self.count_label = ctk.CTkLabel(
            search_frame,
            text="0 records",
            font=config.SMALL_FONT,
            text_color="gray40",
        )
        self.count_label.pack(side="right", padx=10)

        table_frame = ctk.CTkFrame(self)
        table_frame.pack(fill="both", expand=True, padx=30, pady=15)

        self.tree = ttk.Treeview(table_frame, columns=COLUMNS, show="headings")

        widths = {
            "Roll": 80,
            "Name": 150,
            "Branch": 100,
            "Semester": 90,
            "Section": 80,
            "Date": 110,
            "Time": 100,
            "Status": 100,
        }
        for col in COLUMNS:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=widths[col], minwidth=70)

        scrollbar = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

    def _attendance_file(self):
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join("data", "attendance", f"{today}.csv")

    def load_attendance(self):
        """Reload the CSV from disk every time Refresh is clicked."""
        for row in self.tree.get_children():
            self.tree.delete(row)

        file = self._attendance_file()

        if not os.path.exists(file):
            self.df = pd.DataFrame(columns=COLUMNS)
            self.count_label.configure(text="0 records")
            return

        try:
            df = pd.read_csv(file, dtype=str).fillna("")
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            self.df = pd.DataFrame(columns=COLUMNS)
            self.count_label.configure(text="0 records")
            return

        # Ensure old/incomplete CSVs do not break the table.
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        self.df = df[list(COLUMNS)]

        self._populate(self.df)
        self.count_label.configure(text=f"{len(self.df)} records")

    def _populate(self, df):
        for row in self.tree.get_children():
            self.tree.delete(row)

        for _, row in df.iterrows():
            self.tree.insert("", "end", values=[row[col] for col in COLUMNS])

    def search_attendance(self):
        keyword = self.search_entry.get().strip()

        if not keyword:
            self.load_attendance()
            return

        if self.df.empty:
            self._populate(self.df)
            self.count_label.configure(text="0 records")
            return

        result = self.df[
            self.df["Roll"].astype(str).str.contains(keyword, case=False, na=False)
            | self.df["Name"].astype(str).str.contains(keyword, case=False, na=False)
        ]
        self._populate(result)
        self.count_label.configure(text=f"{len(result)} matching records")
