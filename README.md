---
title: Facial-Expression-Video
emoji: 🎭
sdk: gradio
sdk_version: "6.9.0"
python_version: "3.10"
app_file: app.py
---

# FacialCLIP Web: Video & Image Facial Expression Recognition

This project is a web-based implementation of **FacialCLIP**, designed for both real-time video upload processing and static image facial expression recognition. All interfaces and comments have been translated into English for better accessibility.

## 🚀 Live Demo & Model
- **Live Demo:** [Facial-Expression-Video on HF Spaces](https://huggingface.co/spaces/michaelwang66/Facial-Expression-Video)
- **Model Weights:** [michaelwang66/FacialCLIP on HF Hub](https://huggingface.co/michaelwang66/FacialCLIP)

## ✨ Features
- **Video Emotion Recognition:** Upload a video via a web interface and get an annotated video with frame-by-frame expression detection (optimized for CPU).
- **Image Emotion Recognition:** Process single images to detect faces and classify their emotions.
- **Translated Codebase:** Full English support for code, comments, and UI.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/michaelwang66/facialclip_web.git
   cd facialclip_web
   ```

2. **Environment Setup:**
   It is recommended to use the `FacialCLIP` conda environment.
   ```bash
   conda activate FacialCLIP
   # Install dependencies if needed
   pip install -r requirements.txt
   ```

3. **Checkpoints:**
   The web app (`app.py`) will **automatically download** the weights from Hugging Face if they are missing.
   If you want to download them manually for `demo.py`, get them from [here](https://huggingface.co/michaelwang66/FacialCLIP/tree/main).

---

## 🎬 Demo Preview

<p align="center">
  <img src="image/demo.gif" width="45%"/>
  <img src="image/demo2.gif" width="45%"/>
</p>

<p align="center">
  <b>Left:</b> Video Emotion Recognition by seedance &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; <b>Right:</b> Video Emotion Recognition by sora2
</p>
---

## 🖥️ Usage

### 1. Web Application (Video Upload)
The web interface allows you to upload any video file. The system will process it, detect the largest face in each frame, and run the DFER-CLIP model to predict the emotion.

**To start the web app:**
```bash
python app.py
```
After running, open the local URL (usually `http://127.0.0.1:7860`) in your browser.

### 2. Static Image Recognition
For single image processing, you can use `demo.py`. This script will detect a face, align it, and output the predicted emotion with confidence scores.

**To run image recognition:**
```bash
python demo.py --image your_image.jpg
```

---

## 📊 Supported Emotions
The model classifies faces into the following 7 categories:
- Happiness
- Sadness
- Neutral
- Anger
- Surprise
- Disgust
- Fear

---


