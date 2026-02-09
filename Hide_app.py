import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "unified_model_final.pth"
CLASSES = ["clean","stego_js","stego_ps","stego_html","stego_url","stego_eth"]
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- TRANSFORM ----------------
tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# ---------------- MODEL ----------------
class HPF(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([
            [-1,2,-2,2,-1],
            [2,-6,8,-6,2],
            [-2,8,-12,8,-2],
            [2,-6,8,-6,2],
            [-1,2,-2,2,-1]
        ],dtype=torch.float32).unsqueeze(0).unsqueeze(0)/12
        self.w = nn.Parameter(k,requires_grad=False)

    def forward(self,x):
        return F.conv2d(x,self.w.repeat(3,1,1,1),padding=2,groups=3)

class Net(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.hpf = HPF()
        m = models.efficientnet_b2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features,n)
        self.net = m

    def forward(self,x):
        x = self.hpf(x)*0.5 + x
        return self.net(x)

# ---------------- LOAD MODEL ----------------
model = Net(len(CLASSES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------------- PREDICT FUNCTION ----------------
def predict_image(path):
    img = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out,1)[0]
        idx = probs.argmax().item()
        conf = float(probs[idx])

    if CLASSES[idx] != "clean" and conf < 0.60:
        return "CLEAN (low stego confidence)", conf
    else:
        return CLASSES[idx], conf

# ---------------- GUI ----------------
def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files","*.jpg *.png *.jpeg")]
    )
    if file_path:
        label_file.config(text=file_path)
        pred, conf = predict_image(file_path)
        label_result.config(text=f"Prediction: {pred}")
        label_conf.config(text=f"Confidence: {conf:.3f}")

root = tk.Tk()
root.title("Steganography Detection System")
root.geometry("500x300")

title = tk.Label(root, text="Steganography Detection", font=("Arial",16,"bold"))
title.pack(pady=10)

btn = tk.Button(root, text="Select Image", command=open_file, font=("Arial",12))
btn.pack(pady=10)

label_file = tk.Label(root, text="No file selected", wraplength=450)
label_file.pack(pady=5)

label_result = tk.Label(root, text="Prediction: ---", font=("Arial",12,"bold"))
label_result.pack(pady=10)

label_conf = tk.Label(root, text="Confidence: ---", font=("Arial",12))
label_conf.pack(pady=5)

root.mainloop()
