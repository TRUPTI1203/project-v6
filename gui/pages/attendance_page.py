import customtkinter as ctk
from PIL import Image, ImageTk
import config
from controllers.recognize_controller import RecognizeController


class AttendancePage(ctk.CTkFrame):
    """Responsive attendance page with an embedded camera preview.

    The page has two distinct shutdown paths:
    - stop_camera(): user explicitly stops the camera, so the UI is updated.
    - shutdown(): dashboard is navigating away, so no Tk widgets are touched
      after camera shutdown has started.
    This prevents stale PhotoImage/Tcl interpreter errors during page changes.
    """

    def __init__(self, parent):
        super().__init__(parent, fg_color=config.BACKGROUND_COLOR)
        self.controller = RecognizeController(self)
        self._camera_photo = None
        self._poll_job = None
        self._destroyed = False
        self.create_widgets()
        self._schedule_poll()

    def _schedule_poll(self):
        if self._destroyed:
            return
        try:
            if self.winfo_exists():
                self._poll_job = self.after(
                    config.UI_REFRESH_MS,
                    self._poll_ui_and_reschedule,
                )
        except Exception:
            self._poll_job = None

    def _poll_ui_and_reschedule(self):
        self._poll_job = None
        if self._destroyed:
            return
        try:
            if self.winfo_exists():
                self.controller.poll_ui()
                self._schedule_poll()
        except Exception:
            self._poll_job = None

    def create_widgets(self):
        title = ctk.CTkLabel(
            self,
            text="📷 Face Recognition Attendance",
            font=config.TITLE_FONT,
        )
        title.pack(pady=(5, 2))

        subtitle = ctk.CTkLabel(
            self,
            text="Start the camera to recognize students and mark attendance.",
            font=config.SUBTITLE_FONT,
        )
        subtitle.pack(pady=(0, 8))

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=10, pady=5)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        self.camera_card = ctk.CTkFrame(
            main,
            corner_radius=15,
            fg_color="white",
        )
        self.camera_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        ctk.CTkLabel(
            self.camera_card,
            text="Live Camera",
            font=config.HEADING_FONT,
        ).pack(anchor="w", padx=15, pady=(10, 5))

        self.camera_label = self._create_camera_label()

        info_card = ctk.CTkFrame(
            main,
            corner_radius=15,
            fg_color="white",
        )
        info_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        ctk.CTkLabel(
            info_card,
            text="Recognized Student",
            font=config.HEADING_FONT,
        ).pack(anchor="w", padx=20, pady=(20, 15))

        self.status_badge = ctk.CTkLabel(
            info_card,
            text="Waiting for camera...",
            font=config.BUTTON_FONT,
            text_color=config.WARNING_COLOR,
        )
        self.status_badge.pack(anchor="w", padx=20, pady=(0, 15))

        self.name_label = self._detail_label(info_card, "Name", "--")
        self.roll_label = self._detail_label(info_card, "Roll Number", "--")
        self.branch_label = self._detail_label(info_card, "Branch", "--")
        self.semester_label = self._detail_label(info_card, "Semester", "--")
        self.section_label = self._detail_label(info_card, "Section", "--")
        self.confidence_label = self._detail_label(info_card, "Confidence", "--")
        self.time_label = self._detail_label(info_card, "Attendance Time", "--")

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.pack(pady=(2, 8))

        self.start_button = ctk.CTkButton(
            controls,
            text="▶ Start Camera",
            width=180,
            height=42,
            font=config.BUTTON_FONT,
            command=self.start_camera,
        )
        self.start_button.pack(side="left", padx=8)

        self.stop_button = ctk.CTkButton(
            controls,
            text="■ Stop Camera",
            width=180,
            height=42,
            fg_color=config.ERROR_COLOR,
            hover_color="#C62828",
            font=config.BUTTON_FONT,
            command=self.stop_camera,
        )
        self.stop_button.pack(side="left", padx=8)

    def _create_camera_label(self):
        label = ctk.CTkLabel(
            self.camera_card,
            text="📷 Camera is stopped",
            font=config.NORMAL_FONT,
            fg_color="#E5E7EB",
            corner_radius=10,
        )
        label.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return label

    def _reset_camera_preview(self):
        """Replace the camera label instead of clearing a possibly stale PhotoImage.

        CustomTkinter/Tkinter can raise TclError when a PhotoImage belonging to a
        previous widget/interpreter no longer exists. Recreating the small preview
        label avoids calling configure() on that stale image.
        """
        old_label = getattr(self, "camera_label", None)
        try:
            if old_label is not None and old_label.winfo_exists():
                old_label.destroy()
        except Exception:
            pass

        self._camera_photo = None

        if not self._destroyed:
            self.camera_label = self._create_camera_label()

    def _detail_label(self, parent, label, value):
        widget = ctk.CTkLabel(
            parent,
            text=f"{label} : {value}",
            font=config.NORMAL_FONT,
            anchor="w",
        )
        widget.pack(fill="x", padx=20, pady=5)
        return widget

    def start_camera(self):
        if self._destroyed:
            return

        self.start_button.configure(state="disabled", text="● Camera Running")
        self.status_badge.configure(
            text="Scanning...",
            text_color=config.PRIMARY_COLOR,
        )
        self.controller.start()

    def stop_camera(self):
        """User-initiated camera stop. Safe to call repeatedly."""
        if self._destroyed:
            return

        self.controller.stop()

        try:
            self.start_button.configure(
                state="normal",
                text="▶ Start Camera",
            )
            self.status_badge.configure(
                text="Camera stopped",
                text_color=config.WARNING_COLOR,
            )
        except Exception:
            return

        # Recreate the preview widget rather than configuring image=None.
        # This avoids the TclError: image "pyimageXX" doesn't exist.
        self._reset_camera_preview()

    def shutdown(self):
        """Stop all background work before Dashboard destroys this page.

        IMPORTANT: this method intentionally does not configure any Tk widget.
        Dashboard may be in the process of replacing/destroying the page.
        """
        if self._destroyed:
            return

        self._destroyed = True

        poll_job = getattr(self, "_poll_job", None)
        self._poll_job = None
        if poll_job is not None:
            try:
                self.after_cancel(poll_job)
            except Exception:
                pass

        try:
            self.controller.stop()
        except Exception:
            pass

        self._camera_photo = None

    def show_camera_frame(self, frame):
        if self._destroyed:
            return

        try:
            if not self.camera_label.winfo_exists():
                return

            rgb = frame[:, :, ::-1]
            image = Image.fromarray(rgb)
            image.thumbnail((620, 350), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(image=image)
            self._camera_photo = photo
            self.camera_label.configure(image=photo, text="")
        except Exception:
            # Ignore a frame that arrives while the page is being replaced.
            pass

    def update_student(self, info, confidence, status):
        if self._destroyed:
            return

        try:
            if info is None:
                self.status_badge.configure(
                    text=status,
                    text_color=config.ERROR_COLOR,
                )
                return

            self.name_label.configure(text=f"Name : {info['name']}")
            self.roll_label.configure(text=f"Roll Number : {info['roll']}")
            self.branch_label.configure(text=f"Branch : {info['branch']}")
            self.semester_label.configure(text=f"Semester : {info['semester']}")
            self.section_label.configure(text=f"Section : {info['section']}")
            self.confidence_label.configure(
                text=f"Confidence : {confidence * 100:.2f}%"
            )

            from datetime import datetime
            self.time_label.configure(
                text=f"Attendance Time : {datetime.now().strftime('%H:%M:%S')}"
            )

            self.status_badge.configure(
                text=status,
                text_color=(
                    config.SUCCESS_COLOR
                    if status == "Attendance Marked"
                    else config.WARNING_COLOR
                ),
            )
        except Exception:
            pass

    def destroy(self):
        """Tkinter-compatible destruction with safe worker shutdown."""
        self.shutdown()
        try:
            super().destroy()
        except Exception:
            pass
