# 🚀 Real-Time Object Detection Using TensorFlow & OpenCV

This project is a **real-time object detection system** that uses **SSD MobileNet**, **TensorFlow**, and **OpenCV**, integrated into a **Tkinter desktop application**.  
It allows users to detect objects live from their camera with **high accuracy and fast processing**, displaying bounding boxes, labels, and timestamps.

---

## 🎯 Features
- 📷 **Real-Time Object Detection** using webcam  
- 🤖 Uses **SSD MobileNet v3** pre-trained on COCO dataset  
- 🖥️ **User-friendly Tkinter GUI**  
- 🟦 Draws **bounding boxes, labels, and confidence scores**  
- 🕒 Displays **date & time** on the video feed  
- ⚙️ Customizable and easy to extend  
- 🔧 Lightweight and works on **low-end systems**  

---

## 🛠️ Technologies Used
- **Python**
- **TensorFlow Object Detection API**
- **OpenCV**
- **Tkinter (GUI)**
- **NumPy & PIL**

---

## 📁 Project Structure (Table Format)

| File / Folder                  | Description                               |
|-------------------------------|-------------------------------------------|
| `frontend.py`                 | Tkinter GUI for the application           |
| `object_detection.py`         | Detection logic using TensorFlow + OpenCV |
| `frozen_inference_graph.pb`   | Pretrained SSD MobileNet model weights    |
| `ssd_mobilenet_v3.pbtxt`      | Model configuration file                  |
| `labels.txt`                  | COCO dataset labels                       |
| `icons/`                      | Folder containing GUI icons               |



---

## ▶️ How to Run

### 1️⃣ Install required packages:
```bash
pip install opencv-python pillow tensorflow matplotlib

2️⃣ Run the GUI:
python frontend.py

3️⃣ Start object detection:
Click Start Camera → live detection begins.

🌟 Future Enhancements
1. Add custom-trained models
2. Record detection logs
3. Export detected frames
4. Integrate with IoT devices or alert systems

📌 Conclusion
This project demonstrates a functional, fast, and efficient real-time object detection system, suitable for applications such as surveillance, automation, smart cameras, and AI vision systems.

