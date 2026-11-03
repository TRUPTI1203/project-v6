import customtkinter as ctk
import config
from tkinter import messagebox

from models.student import Student
from controllers.register_controller import RegisterController


class RegisterPage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color=config.BACKGROUND_COLOR)
        self.controller = RegisterController()
        self.create_widgets()

    def create_widgets(self):
        title = ctk.CTkLabel(self, text="👤 Register New Student", font=config.TITLE_FONT)
        title.pack(pady=(20, 5))

        subtitle = ctk.CTkLabel(
            self,
            text="Enter student details before capturing face images.",
            font=config.SUBTITLE_FONT,
        )
        subtitle.pack(pady=(0, 20))

        form = ctk.CTkFrame(self, width=500, corner_radius=15, fg_color="white")
        form.pack(pady=10, padx=20)

        ctk.CTkLabel(form, text="Student Name", font=config.NORMAL_FONT).pack(
            anchor="w", padx=20, pady=(20, 5)
        )
        self.name_entry = ctk.CTkEntry(
            form, width=400, height=40, placeholder_text="Enter Full Name"
        )
        self.name_entry.pack(padx=20)

        ctk.CTkLabel(form, text="Roll Number", font=config.NORMAL_FONT).pack(
            anchor="w", padx=20, pady=(15, 5)
        )
        self.roll_entry = ctk.CTkEntry(
            form, width=400, height=40, placeholder_text="Enter Roll Number"
        )
        self.roll_entry.pack(padx=20)

        ctk.CTkLabel(form, text="Branch", font=config.NORMAL_FONT).pack(
            anchor="w", padx=20, pady=(15, 5)
        )
        self.branch_combo = ctk.CTkComboBox(
            form, width=400, values=["CSE", "ECE", "EEE", "Mechanical", "Civil"]
        )
        self.branch_combo.pack(padx=20)

        ctk.CTkLabel(form, text="Semester", font=config.NORMAL_FONT).pack(
            anchor="w", padx=20, pady=(15, 5)
        )
        self.semester_combo = ctk.CTkComboBox(
            form, width=400, values=["1", "2", "3", "4", "5", "6", "7", "8"]
        )
        self.semester_combo.pack(padx=20)

        ctk.CTkLabel(form, text="Section", font=config.NORMAL_FONT).pack(
            anchor="w", padx=20, pady=(15, 5)
        )
        self.section_combo = ctk.CTkComboBox(
            form, width=400, values=["A", "B", "C", "D"]
        )
        self.section_combo.pack(padx=20)

        self.register_button = ctk.CTkButton(
            form,
            text="📷 Start Registration",
            width=250,
            height=45,
            font=config.BUTTON_FONT,
            command=self.start_registration,
        )
        self.register_button.pack(pady=30)

        self.progress_label = ctk.CTkLabel(
            self,
            text="",
            font=config.SMALL_FONT,
            text_color=config.TEXT_COLOR,
        )
        self.progress_label.pack(pady=(0, 5))

    def start_registration(self):
        student = Student(
            name=self.name_entry.get(),
            roll=self.roll_entry.get(),
            branch=self.branch_combo.get(),
            semester=self.semester_combo.get(),
            section=self.section_combo.get(),
        )

        valid, message = self.controller.validate(student)
        if not valid:
            messagebox.showerror("Registration", message)
            return

        if self.controller.student_exists(student.roll):
            messagebox.showerror("Registration", "Student already registered.")
            return

        self.register_button.configure(state="disabled", text="⏳ Capturing...")
        self.progress_label.configure(
            text="Camera is running. Follow the pose instructions in the camera window."
        )

        self.controller.register_student_async(student, self._registration_finished)

    def _registration_finished(self, success, message):
        # Callback runs on the worker thread; move GUI work back to Tk's main thread.
        self.after(0, lambda: self._show_registration_result(success, message))

    def _show_registration_result(self, success, message):
        self.register_button.configure(state="normal", text="📷 Start Registration")
        self.progress_label.configure(text="")

        if success:
            messagebox.showinfo("Registration", message)
        else:
            messagebox.showerror("Registration", message)
