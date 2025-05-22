import matplotlib.pyplot as plt
import struct
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import coremltools as ct
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np


def load_images_torch(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read the rest of the bytes
        raw_data = f.read(num * rows * cols)
        
        # Create a 1D uint8 tensor from bytes
        data_tensor = torch.tensor(list(raw_data), dtype=torch.uint8)
        
        # Reshape to [num, rows, cols]
        images = data_tensor.view(num, rows, cols)
        
        return images


def load_labels_torch(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        raw_data = f.read(num)  # read 'num' bytes
        labels = torch.tensor(list(raw_data), dtype=torch.uint8)
        return labels

def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


class numberRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # After conv and pooling: feature map size will be 7x7 with 32 channels (same as before)
        self.feature_size = 32 * 7 * 7
        
        # Instead of flattening directly, reshape feature map into sequence of tokens for transformer
        # We'll treat each spatial position as a token with 32 features
        
        self.embedding_dim = 32
        self.sequence_length = 7 * 7  # 49 tokens
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Classification head
        self.fc = nn.Linear(self.embedding_dim, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # x shape: (batch_size, channels=32, height=7, width=7)
        
        # Reshape for transformer: (sequence_length, batch_size, embedding_dim)
        # First flatten height and width into sequence length, then permute
        x = x.flatten(2)  # (batch_size, channels=32, 49)
        x = x.permute(2, 0, 1)  # (49, batch_size, 32)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Take the mean of the sequence output (simple pooling)
        x = x.mean(dim=0)  # (batch_size, embedding_dim)
        
        # Classification layer
        x = self.fc(x)  # (batch_size, 10)
        
        return x

# Replace with your file paths


train_images = (load_images_torch("/Users/srivatsansureshbabu/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/train-images-idx3-ubyte/train-images-idx3-ubyte")).float()/255.0
train_labels = (load_labels_torch("/Users/srivatsansureshbabu/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/train-labels-idx1-ubyte/train-labels-idx1-ubyte")).long()

test_images = (load_images_torch("/Users/srivatsansureshbabu/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")).float()/255.0
test_labels = (load_labels_torch("/Users/srivatsansureshbabu/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")).long()

train_dataset = TensorDataset(train_images,train_labels)
test_dataset = TensorDataset(test_images,test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# normalization of data
model = numberRecognizer()


if torch.backends.mps.is_available():
    print("✅ MPS is available!")
else:
    print("❌ MPS not available.")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


lossFunctionOutput = nn.CrossEntropyLoss().to(device)


model = numberRecognizer().to(device)
model.load_state_dict(torch.load("numberRecognizer.pth"))
model.eval()

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
    global drawing_array
    img_resized = image.resize((28, 28), resample=Image.Resampling.LANCZOS)
    arr = np.array(img_resized)
    drawing_array = arr
    print("Array shape:", arr.shape)
    print(arr)
    root.destroy()

canvas.bind('<B1-Motion>', paint)
canvas.bind('<ButtonRelease-1>', reset)

btn_save = tk.Button(root, text="Save & Quit", command=save_and_quit)
btn_save.pack()

root.mainloop()

tensor = torch.from_numpy(drawing_array).float()/255.0

# Disable gradient calculation for inference
with torch.no_grad():
    output = model(tensor.unsqueeze(0).unsqueeze(0).to(device))

# For classification: get the predicted class
predicted_class = torch.argmax(output, dim=1)

print("Predicted class:", predicted_class.item())# total = 0
