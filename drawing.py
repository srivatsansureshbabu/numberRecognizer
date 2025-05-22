import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

width, height = 280, 280
pen_width = 15

root = tk.Tk()
root.title("Draw with Mouse")

canvas = tk.Canvas(root, width=width, height=height, bg='black')
canvas.pack()

# PIL image and drawing context to save drawing
image = Image.new("L", (width, height), 0)  # black background
draw = ImageDraw.Draw(image)

last_x, last_y = None, None

def paint(event):
    global last_x, last_y
    x, y = event.x, event.y
    if last_x and last_y:
        canvas.create_line(last_x, last_y, x, y, fill='white', width=pen_width, capstyle=tk.ROUND, smooth=True)
        draw.line([last_x, last_y, x, y], fill=255, width=pen_width)
    last_x, last_y = x, y

def reset(event):
    global last_x, last_y
    last_x, last_y = None, None

def save_and_quit():
    img_resized = image.resize((28, 28), resample=Image.Resampling.LANCZOS)
    arr = np.array(img_resized)
    print("Array shape:", arr.shape)
    print(arr)
    root.destroy()

canvas.bind('<B1-Motion>', paint)
canvas.bind('<ButtonRelease-1>', reset)

btn_save = tk.Button(root, text="Save & Quit", command=save_and_quit)
btn_save.pack()

root.mainloop()
