import os
import sys

#  --- [1] Forced shielding of underlying interference logs ---
os.environ["ORT_LOGGING_LEVEL"] = "3" 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gradio as gr
import torch
import cv2
import numpy as np
from collections import deque
from PIL import Image

#  --- [2] Manually define the RecorderMeter class ---
class RecorderMeter(object):
    def __init__(self, total_epoch):
        self.total_epoch = total_epoch
        self.epoch_losses = None
        self.epoch_errors = None

torch.serialization.add_safe_globals([RecorderMeter])

#  --- [3] Initialize model ---
print(">>> Step 1/3: Loading RetinaFace...")
from models.Generate_Model import GenerateModel
from clip import clip
import insightface
from insightface.app import FaceAnalysis

#  Force the use of a single provider to reduce thread conflicts
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print(">>> [OK] RetinaFace loaded successfully!")

print(">>> Step 2/3: Loading DFER-CLIP...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLIP_model, _ = clip.load("ViT-B/32", device=device)
CLIP_model = CLIP_model.float().eval()

class DummyArgs:
    contexts_number = 8
    class_token_position = "end"
    class_specific_contexts = 'True'
    load_and_tune_prompt_learner = 'False'
    temporal_layers = 1
    text_type = 'class_descriptor'

args = DummyArgs()
from models.Text import class_descriptor_7
model = GenerateModel(input_text=class_descriptor_7, clip_model=CLIP_model, args=args)

#  weight path
PTH_PATH = "./checkpoint/DFEW-2508291647mydfewexp-set1-model_best.pth"
checkpoint = torch.load(PTH_PATH, map_location=device, weights_only=False)
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model = model.to(device).float().eval()
print(">>> [OK] DFER-CLIP weights loaded successfully!")

#  --- [4] Improved version of reasoning logic (click once to get the result) ---
EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]

def predict_emotion(img):
    if img is None:
        return "Please turn on the camera first, take a photo and then click Submit"

    print("\n>>> Receive front-end data and start processing...")
    # gr.Image(type="numpy") -> img is np.ndarray RGB uint8
    img_rgb = img
    if not isinstance(img_rgb, np.ndarray):
        img_rgb = np.array(img_rgb)

    #  RGB -> BGR for insightface
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    faces = face_app.get(img_bgr)
    if len(faces) == 0:
        print(">>> No face detected")
        return "No face detected, please face the camera, have sufficient lighting, and put your face in the middle of the screen"

    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    aimg = insightface.utils.face_align.norm_crop(img_bgr, landmark=face.kps, image_size=224)

    #  BGR -> RGB refeed the model (most PyTorch models are trained on RGB)
    aimg_rgb = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)

    img_tensor = torch.from_numpy(aimg_rgb).permute(2, 0, 1).float() / 255.0  # [C,H,W]

    #  Copy 16 frames
    T = 16
    frames = img_tensor.unsqueeze(0).repeat(T, 1, 1, 1)  # [T,C,H,W]

    #  ✅ Common timing model expectations [B,C,T,H,W]
    input_tensor = frames.permute(1,0,2,3).unsqueeze(0).to(device)  # [1,C,T,H,W]

    print(">>> Input tensor shape:", tuple(input_tensor.shape))

    print(">>> Model inference in progress...")
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        idx = torch.argmax(prob, dim=1).item()

    conf = float(prob[0, idx].item())
    result = f"Recognition result: {EMOTIONS[idx]} (Confidence: {conf*100:.1f}%)"
    print(f">>> Inference completed: {result}")
    return result


interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="numpy"),  #  ✅ Fixed to numpy
    outputs="text",
    live=False,  #  Manual photo + Submit
    title="DFER-CLIP demonstration (photo recognition)",
    description="Steps: Turn on the camera → Click Take Photo → Click Submit → Get the expression results"
)
#  Force port 7860 to be specified
interface.launch(server_name="0.0.0.0", server_port=7860, share=True)