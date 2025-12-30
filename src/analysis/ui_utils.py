import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class UIUtils:
    @staticmethod
    def setup_window(title, width=1200, height=800):
        root = tk.Tk()
        root.title(title)
        root.geometry(f"{width}x{height}")
        return root

    @staticmethod
    def create_panel(root, callback, btn_txt="Generate"):
        frame = tk.Frame(root, bg="#eee", pady=10)
        frame.pack(side=tk.TOP, fill=tk.X)
        btn = tk.Button(frame, text=btn_txt, command=callback, bg="#4CAF50", fg="white", font=("Arial", 12))
        btn.pack(side=tk.LEFT, padx=20)
        lbl = tk.Label(frame, text="Ready", bg="#eee", font=("Consolas", 10))
        lbl.pack(side=tk.LEFT, padx=20)
        return lbl

    @staticmethod
    def np_to_tk(arr, size=None):
        arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(arr_u8, mode='L')
        if size: img = img.resize((size, size), Image.Resampling.NEAREST)
        return ImageTk.PhotoImage(img)

    @staticmethod
    def embed_plot(root, subplots=(1, 3)):
        fig = Figure(figsize=(15, 5), dpi=100)
        axs = fig.subplots(*subplots)
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return fig, axs, canvas