"""
==========================================
Face Recognition Attendance System
Main Application
==========================================
"""

import customtkinter as ctk

import config
from gui.dashboard import Dashboard


class FaceRecognitionApp(ctk.CTk):
    """
    Main Application Window
    """

    def __init__(self):
        super().__init__()

        self.configure_window()

        self.dashboard = Dashboard(self)

    def configure_window(self):
        """
        Configure the main application window.
        """

        self.title(config.WINDOW_TITLE)

        self.geometry(
            f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}"
        )

        self.minsize(
            config.WINDOW_WIDTH,
            config.WINDOW_HEIGHT
        )

        self.configure(
            fg_color=config.BACKGROUND_COLOR
        )


if __name__ == "__main__":

    ctk.set_appearance_mode("light")

    ctk.set_default_color_theme("blue")

    app = FaceRecognitionApp()

    app.mainloop()