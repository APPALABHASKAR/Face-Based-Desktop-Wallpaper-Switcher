# Face-Based-Desktop-Wallpaper-Switcher

This project uses real-time face recognition via OpenCV to dynamically change your desktop wallpaper based on whether an authorized face is detected. It is ideal for basic personalization or low-level security awareness on personal machines.

🧠 How It Works
Captures video from your webcam in real time

Detects and recognizes faces using the LBPH algorithm

Compares with trained face data to determine if the user is authorized

Automatically sets:

✅ Authorized wallpaper if the user is recognized

🚫 Unauthorized wallpaper otherwise

🔧 Features
🎥 Real-time webcam video processing

🧑‍💻 Face dataset generation and training with OpenCV

💾 LBPH face recognition with saved model

🖼️ Wallpaper changing via Windows API (ctypes)

👤 Multi-user recognition support

🔒 Simple privacy indication with visual cues

📁 Project Structure
📂 python projects/
├── data/                    # Stores face samples
│   ├── user.1.1.jpg
│   └── ...
├── My_wall.jpg              # Wallpaper for recognized users
├── unauthorised.jpg         # Wallpaper for unrecognized access
├── classifier.xml           # Trained face recognition model
└── Wallpaper_change.py      # Main script
▶️ Usage Instructions
Collect Face Samples (optional, run once):

python
take_user_face_dataset()
Train the Classifier:

python
train_classifier("data")
Run the Recognition & Wallpaper Switching:

bash
python Wallpaper_change.py
Press q to quit the camera stream.

🧰 Requirements
Install dependencies with:

bash
pip install opencv-python Pillow numpy
Also ensure:

You are on Windows OS (uses ctypes for wallpaper control)

Webcam access is enabled

haarcascade_frontalface_default.xml is in the working directory

📌 Notes
Default user IDs are set as 1 and 2. You can customize IDs in the dataset.

The script assumes wallpapers are located at:

C:\python projects\My_wall.jpg

C:\python projects\unauthorised.jpg
