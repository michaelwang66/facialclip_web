import os
import sys

# Suppress underlying interference logs
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from collections import deque
import gradio as gr
from huggingface_hub import hf_hub_download

# ----- Must define RecorderMeter before torch.load -----
class RecorderMeter(object):
    def __init__(self, total_epoch):
        self.total_epoch = total_epoch
        self.epoch_losses = None
        self.epoch_errors = None

# add_safe_globals is an API of PyTorch 2.4+; older versions load with weights_only=False and don't need it
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([RecorderMeter])

# ---- Parse command line arguments ----
class Args:
    checkpoint = './checkpoint/DFEW-2508291647mydfewexp-set1-model_best.pth'
    contexts_number = 8
    class_token_position = 'end'
    class_specific_contexts = 'True'
    load_and_tune_prompt_learner = 'False'
    temporal_layers = 1
    text_type = 'class_descriptor'

args = Args()

EMOTIONS = ["Happiness", "Sadness", "Neutral", "Anger", "Surprise", "Disgust", "Fear"]

def load_model(device):
    from models.Generate_Model import GenerateModel
    from models.clip import clip
    from models.Text import class_descriptor_7

    print(">>> [1/2] Loading CLIP model...")
    CLIP_model, _ = clip.load("ViT-B/32", device=device)
    CLIP_model = CLIP_model.float().eval()
    print(">>> [OK] CLIP loaded successfully")

    print(">>> [2/2] Loading DFER-CLIP weights...")
    model = GenerateModel(input_text=class_descriptor_7, clip_model=CLIP_model, args=args)

    if not os.path.exists(args.checkpoint):
        print(f">>> Checkpoint not found at {args.checkpoint}. Attempting to download from Hugging Face...")
        try:
            # You will need to replace 'your-username/your-model-name' with your actual HF repo
            # and 'filename.pth' with your file name.
            repo_id = os.getenv("HF_MODEL_REPO", "michaelwang66/FacialCLIP") 
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename="DFEW-set1-model_best.pth")
            args.checkpoint = checkpoint_path
            print(f">>> Downloaded checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"[Error] Failed to download checkpoint: {e}")
            raise FileNotFoundError(f"Cannot find or download weights file.")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model = model.to(device).float().eval()
    print(">>> [OK] Weights loaded successfully\n")
    return model

def load_face_detector(device):
    from facenet_pytorch import MTCNN
    print(">>> Loading face detector (MTCNN, pure PyTorch)...")
    mtcnn = MTCNN(
        image_size=224,
        margin=20,
        keep_all=False,
        post_process=False,   # Return uint8 [0,255] Tensor
        device=device,
    )
    print(">>> [OK] Face detector loaded successfully\n")
    return mtcnn

# Check CUDA availability
major, minor = 0, 0
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    sm = major * 10 + minor
    supported_sms = {50, 60, 70, 75, 80, 86, 89, 90, 100, 120}
    if sm in supported_sms:
        device = torch.device("cuda:0")
    else:
        print(f">>> Warning: Current GPU sm_{sm} is not compatible with PyTorch, automatically switching to CPU")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

print(f">>> Using device: {device}\n")

print(">>> Initializing models globally (this might take a moment)...")
mtcnn = load_face_detector(device)
model = load_model(device)
print(">>> Initialization complete. Ready for video processing.")

def process_video(video_path):
    if not video_path:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps: # Check for NaN or 0
        fps = 30.0
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    import imageio
    out_path = "output_video.mp4"
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=None)

    clip_buf = deque(maxlen=16)
    last_bbox = None
    last_label = "Detecting..."
    last_score = 0.0

    SKIP_FRAMES = 3  # Only detect and predict every 3 frames to save CPU
    print(f">>> Processing video: {video_path} (Frame skipping: {SKIP_FRAMES})")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Only run detection/inference every SKIP_FRAMES
        should_process = (frame_count == 1) or (frame_count % SKIP_FRAMES == 0)
        
        if should_process:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # Detect face boxes and probabilities
            boxes, probs = mtcnn.detect(pil_img)
            
            if boxes is not None and len(boxes) > 0:
                last_bbox = [int(v) for v in boxes[0]]
                
                # Extract face tensor
                face_tensor = mtcnn.extract(pil_img, boxes, None)
                if face_tensor is not None:
                    img_tensor = face_tensor.float() / 255.0  # [3, 224, 224]
                    
                    clip_buf.append(img_tensor)

                    if len(clip_buf) == 16:
                        frames = torch.stack(list(clip_buf), dim=0) # [16, 3, 224, 224]
                        input_tensor = frames.unsqueeze(0).to(device) # [1, 16, 3, 224, 224]

                        # Inference
                        with torch.no_grad():
                            output = model(input_tensor)
                            prob = torch.softmax(output, dim=1)[0]
                            idx = torch.argmax(prob).item()
                            
                        last_label = EMOTIONS[idx]
                        last_score = float(prob[idx]) * 100
        # Else: use last_bbox, last_label from previous processed frame

        # Draw box and label if available
        if last_bbox is not None:
            x1, y1, x2, y2 = last_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{last_label} ({last_score:.1f}%)"
            cv2.putText(frame, text, (x1, max(20, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        frame_out_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_out_rgb)
        
        if frame_count % 30 == 0:
            print(f">>> Processed {frame_count} frames...")

    cap.release()
    writer.close()
    print(">>> Video processing complete!")
    return out_path

# Gradio Interface
with gr.Blocks(title="DFER-CLIP Video Emotion Recognition") as iface:
    gr.Markdown("# Video Real-time Facial Expression Recognition")
    gr.Markdown("Upload a video and the system will process it frame by frame to detect the facial expression of the largest face in the frame.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video", sources=["upload"])
            process_btn = gr.Button("Process Video")
        with gr.Column():
            video_output = gr.Video(label="Output Video with Emotion")
            
    process_btn.click(fn=process_video, inputs=video_input, outputs=video_output)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
