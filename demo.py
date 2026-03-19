import os
import sys

#  Block underlying interference logs
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import torch
import cv2
import numpy as np
from PIL import Image

#  ----- RecorderMeter must be defined before torch.load -----
class RecorderMeter(object):
    def __init__(self, total_epoch):
        self.total_epoch = total_epoch
        self.epoch_losses = None
        self.epoch_errors = None

#  add_safe_globals is an API for PyTorch 2.4+; older versions are loaded with weights_only=False and do not require it
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([RecorderMeter])

#  ---- Parse command line parameters ----
def get_args():
    parser = argparse.ArgumentParser(description="DFER-CLIP single picture expression recognition")
    parser.add_argument('--image', type=str, default='ne.png',
                        help='Enter the image path (default: ku.png)')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoint/DFEW-2508291647mydfewexp-set1-model_best.pth',
                        help='Model weight path')
    parser.add_argument('--contexts-number', type=int, default=8)
    parser.add_argument('--class-token-position', type=str, default='end')
    parser.add_argument('--class-specific-contexts', type=str, default='True')
    parser.add_argument('--load-and-tune-prompt-learner', type=str, default='False')
    parser.add_argument('--temporal-layers', type=int, default=1)
    parser.add_argument('--text-type', type=str, default='class_descriptor')
    return parser.parse_args()


EMOTIONS = ["happiness", "sadness", "neutral", "anger", "surprise", "disgust", "fear"]
EMOTIONS_ZH = ["happy", "sad", "neutral", "anger", "surprise", "disgust", "fear"]


def load_model(args, device):
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
        print(f"[Error] Weights file not found: {args.checkpoint}")
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model = model.to(device).float().eval()
    print(">>> [OK] Weights loaded successfully\n")
    return model


def load_face_detector(device):
    from facenet_pytorch import MTCNN
    print(">>> Loading face detector (MTCNN, pure PyTorch)...")
    #  keep_all=False only returns the face with the highest confidence, image_size=224 directly crops to the target size
    mtcnn = MTCNN(
        image_size=224,
        margin=20,
        keep_all=False,
        post_process=False,   #  Return uint8 [0,255] Tensor to facilitate subsequent processing
        device=device,
    )
    print(">>> [OK] Face detector loaded successfully\n")
    return mtcnn


def predict(image_path, model, mtcnn, device):
    #  Read pictures
    if not os.path.exists(image_path):
        print(f"[Error] Image does not exist: {image_path}")
        sys.exit(1)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[Error] Unable to read image: {image_path}")
        sys.exit(1)

    print(f">>> Input image: {image_path} Size: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    #  BGR -> RGB PIL, fed to MTCNN
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    #  Face detection + cropping alignment (post_process=False -> uint8 Tensor [3,224,224])
    face_tensor = mtcnn(pil_img)   #  None means no face detected
    if face_tensor is None:
        print("[Result] No face detected, please make sure there is a clear face in the picture")
        return

    print(">>> Face detected, cropped and aligned to 224x224")

    #  uint8 [0,255] -> float [0,1], copied to 16 frames
    img_tensor = face_tensor.float() / 255.0   # [3,224,224]
    T = 16
    frames = img_tensor.unsqueeze(0).repeat(T, 1, 1, 1)           # [T,3,224,224]
    #  GenerateModel.forward() expects [N, T, C, H, W] (n,t,c,h,w = image.shape in the code)
    input_tensor = frames.unsqueeze(0).to(device)  # [1,T,3,224,224]

    #  reasoning
    print(">>> Reasoning...")
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)[0]
        idx = torch.argmax(prob).item()

    #  Print results
    print("\n" + "=" * 45)
    print(f"  Expression recognition results: {EMOTIONS_ZH[idx]} / {EMOTIONS[idx]}")
    print(f"  Confidence: {float(prob[idx]) * 100:.1f}%")
    print("=" * 45)
    print("\nScores for each category:")
    for i, (zh, en) in enumerate(zip(EMOTIONS_ZH, EMOTIONS)):
        bar = "█" * int(float(prob[i]) * 30)
        print(f"  {zh:4s}/{en:8s}: {bar:<30s} {float(prob[i])*100:5.1f}%")
    print()

    #  Save the cropped face
    face_np = face_tensor.permute(1, 2, 0).byte().numpy()          # [224,224,3] RGB
    face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    face_save_path = image_path.rsplit('.', 1)[0] + '_face_aligned.jpg'
    cv2.imwrite(face_save_path, face_bgr)
    print(f">>> Save the aligned face image to: {face_save_path}")


def main():
    args = get_args()

    #  Check if CUDA is available and the GPU is compatible with current PyTorch (RTX 5090 / sm_120 requires special version)
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        sm = major * 10 + minor
        #  PyTorch precompiled CUDA kernel supports up to sm_90; sm_120 (RTX 5090) requires nightly
        supported_sms = {50, 60, 70, 75, 80, 86, 89, 90}
        if sm in supported_sms:
            device = torch.device("cuda:0")
        else:
            print(f">>> Warning: The current GPU sm_{sm} is incompatible with PyTorch and will automatically switch to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f">>> Device used: {device}\n")

    mtcnn = load_face_detector(device)
    model = load_model(args, device)
    predict(args.image, model, mtcnn, device)


if __name__ == '__main__':
    main()
