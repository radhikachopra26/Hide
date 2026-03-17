🔐 Steganography Detection System (Deep Learning + CNN)

A GUI-based Steganography Detection System built using PyTorch and EfficientNet-B2, capable of detecting hidden data in images across multiple steganography techniques.

This project combines High-Pass Filtering (HPF) with a CNN-based classifier to identify whether an image is clean or contains steganographic content.

🚀 Features

📂 Select image via GUI (Tkinter)

🧠 Deep Learning model (EfficientNet-B2)

🔍 High-Pass Filter preprocessing for noise extraction

🎯 Multi-class classification:

clean

stego_js

stego_ps

stego_html

stego_url

stego_eth

📊 Confidence score output

⚡ GPU support (if available)

🛠️ Tech Stack

Python 3.x

PyTorch

Torchvision

Tkinter (GUI)

PIL (Image Processing)

🧠 Model Architecture

Custom HPF Layer (Noise enhancement)

EfficientNet-B2

Final fully connected layer adapted for 6 classes

Input Image → HPF Filter → EfficientNet-B2 → Classification
📁 Project Structure
├── unified_model_final.pth   # Trained model
├── main.py                   # Main GUI application
├── README.md
⚙️ Installation
git clone https://github.com/your-username/steganography-detector.git
cd steganography-detector

pip install torch torchvision pillow
▶️ Usage

Run the application:

python main.py
Steps:

Click "Select Image"

Choose an image (.jpg, .png, .jpeg)

View:

Prediction result

Confidence score

📊 Output Example
Prediction: stego_html
Confidence: 0.872

⚠️ If confidence < 0.60 for stego classes → marked as:

CLEAN (low stego confidence)
🔍 How It Works

Image is resized to 224x224

Normalization applied

High-Pass Filter (HPF) extracts hidden noise patterns

Passed through EfficientNet-B2

Softmax generates probabilities

Final prediction based on highest confidence
