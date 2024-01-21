import os
import tkinter as tk
from PIL import Image, ImageTk

from interface.view_data_panel import ViewData


class TitleScreen:
    def __init__(self, root):
        self.title_screen_canvas = None
        self.view_data = None

        self.window = root
        self.setup_window()

    def setup_window(self):
        self.window.geometry("1200x900")
        self.window.title("AI Project")
        self.window.resizable(False, False)

        self.title_screen_canvas = tk.Canvas(self.window, bd=0, highlightthickness=0, relief='ridge')
        self.title_screen_canvas.pack(fill=tk.BOTH, expand=True)

        image = Image.open(os.path.join("images", "title_screen.png"))
        photo = ImageTk.PhotoImage(image)

        self.title_screen_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.title_screen_canvas.image = photo
        self.init_button()

    def init_button(self):
        button_width = 20
        button_height = 3

        button_font = ("Arial", 14)

        button = tk.Button(self.window, text="Începe",
                           command=lambda: self.on_button_click(),
                           width=button_width, height=button_height, borderwidth=2, highlightthickness=0,
                           font=button_font)
        self.title_screen_canvas.create_window(800, 450, window=button, anchor='nw')

    def on_button_click(self):
        self.title_screen_canvas.destroy()
        self.view_data = ViewData(self.window)
