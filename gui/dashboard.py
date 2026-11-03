"""
==========================================
Dashboard
Face Recognition Attendance System
==========================================
"""

import customtkinter as ctk
import config

# Import pages
from gui.pages.home_page import HomePage
from gui.pages.register_page import RegisterPage
from gui.pages.attendance_page import AttendancePage
from gui.pages.reports_page import ReportsPage


class Dashboard(ctk.CTkFrame):

    def __init__(self, master):
        super().__init__(master, fg_color=config.BACKGROUND_COLOR)

        self.pack(fill="both", expand=True)

        self.current_page = None

        self.create_layout()

        self.show_page(HomePage)

    # ----------------------------------------------------
    # Main Layout
    # ----------------------------------------------------

    def create_layout(self):

        self.create_header()

        self.create_body()

        self.create_footer()

    # ----------------------------------------------------
    # Header
    # ----------------------------------------------------

    def create_header(self):

        self.header = ctk.CTkFrame(
            self,
            height=config.HEADER_HEIGHT,
            fg_color="white",
            corner_radius=0
        )

        self.header.pack(fill="x")

        self.header.pack_propagate(False)

        title = ctk.CTkLabel(
            self.header,
            text="🎓 Face Recognition Attendance System",
            font=config.TITLE_FONT
        )

        title.pack(anchor="w", padx=20, pady=(12, 0))

        subtitle = ctk.CTkLabel(
            self.header,
            text="Artificial Intelligence | Computer Vision | Deep Learning",
            font=config.SUBTITLE_FONT
        )

        subtitle.pack(anchor="w", padx=22)

    # ----------------------------------------------------
    # Body
    # ----------------------------------------------------

    def create_body(self):

        self.body = ctk.CTkFrame(
            self,
            fg_color=config.BACKGROUND_COLOR
        )

        self.body.pack(fill="both", expand=True)

        self.create_sidebar()

        self.create_content()

    # ----------------------------------------------------
    # Sidebar
    # ----------------------------------------------------

    def create_sidebar(self):

        self.sidebar = ctk.CTkFrame(
            self.body,
            width=config.SIDEBAR_WIDTH,
            fg_color=config.SIDEBAR_COLOR,
            corner_radius=0
        )

        self.sidebar.pack(side="left", fill="y")

        self.sidebar.pack_propagate(False)

        title = ctk.CTkLabel(
            self.sidebar,
            text="Navigation",
            text_color="white",
            font=config.HEADING_FONT
        )

        title.pack(pady=25)

        self.create_nav_button(
            "🏠 Dashboard",
            lambda: self.show_page(HomePage)
        )

        self.create_nav_button(
            "👤 Register Student",
            lambda: self.show_page(RegisterPage)
        )

        self.create_nav_button(
            "📷 Attendance",
            lambda: self.show_page(AttendancePage)
        )

        self.create_nav_button(
            "📊 Reports",
            lambda: self.show_page(ReportsPage)
        )

        self.create_nav_button(
            "❌ Exit",
            self.master.destroy
        )

    # ----------------------------------------------------
    # Sidebar Buttons
    # ----------------------------------------------------

    def create_nav_button(self, text, command):

        btn = ctk.CTkButton(
            self.sidebar,
            text=text,
            width=190,
            height=45,
            font=config.BUTTON_FONT,
            command=command
        )

        btn.pack(pady=8)

    # ----------------------------------------------------
    # Content Area
    # ----------------------------------------------------

    def create_content(self):

        self.content = ctk.CTkFrame(
            self.body,
            fg_color=config.BACKGROUND_COLOR
        )

        self.content.pack(
            side="left",
            fill="both",
            expand=True,
            padx=20,
            pady=20
        )

    # ----------------------------------------------------
    # Footer
    # ----------------------------------------------------

    def create_footer(self):

        self.footer = ctk.CTkFrame(
            self,
            height=config.FOOTER_HEIGHT,
            fg_color="white",
            corner_radius=0
        )

        self.footer.pack(fill="x")

        self.footer.pack_propagate(False)

        self.status = ctk.CTkLabel(
            self.footer,
            text="🟢 Ready",
            font=config.SMALL_FONT
        )

        self.status.pack(side="left", padx=20)

        version = ctk.CTkLabel(
            self.footer,
            text="Version 1.0",
            font=config.SMALL_FONT
        )

        version.pack(side="right", padx=20)

    # ----------------------------------------------------
    # Page Switching
    # ----------------------------------------------------

    def show_page(self, page_class):
        # Stop any active camera before destroying the current page.
        # The Attendance controller waits for its worker to release the webcam,
        # which prevents the Register page from inheriting a locked camera.
        if self.current_page is not None:
            shutdown = getattr(self.current_page, "shutdown", None)
            if callable(shutdown):
                shutdown()
            else:
                stop = getattr(self.current_page, "stop_camera", None)
                if callable(stop):
                    stop()
            self.current_page.destroy()

        self.current_page = page_class(self.content)
        self.current_page.pack(fill="both", expand=True)

        page_name = page_class.__name__
        status_text = {
            "HomePage": "🟢 Dashboard",
            "RegisterPage": "🟢 Registration",
            "AttendancePage": "🟢 Attendance",
            "ReportsPage": "🟢 Reports",
        }.get(page_name, "🟢 Ready")
        self.status.configure(text=status_text)