# import cv2
# import numpy as np
# from collections import deque, defaultdict

# from retinaface import RetinaFace  # pip install retina-face
# import torch

#  # ---- 1) Load DFER-CLIP (you need to adapt according to the warehouse structure) ----
# # from models.Generate_Model import GenerateModel
# # from clip import clip
#  # This is represented by a placeholder function
# class DferClipWrapper:
#     def __init__(self, ckpt_path: str, device="cuda"):
#         self.device = device
#         # CLIP_model, _ = clip.load("ViT-B/32", device="cpu")
#         # model = GenerateModel(input_text=..., clip_model=CLIP_model, args=...)
#         # model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
#         # self.model = model.to(device).eval()
#         self.model = None

#        
#         self.class_names = ["angry","disgust","fear","happy","sad","surprise","neutral"]

#     @torch.no_grad()
#     def predict_clip(self, faces_16):  # faces_16: list of 16 aligned face images (HWC BGR or RGB)
#         """
#          Return: (label, score)
#        
#         1) BGR->RGB
#          2) resize/crop to 224
#          3) Normalize to mean/std required by CLIP
#          4) stack into tensor: (1, 16, 3, 224, 224)
#         5) self.model(images) -> logits
#         """
#         # ---- placeholder ----
#         return "neutral", 0.0


#  # ---- 2) Simple face selection (single scene: choose the largest face) ----
# def pick_largest_face(resp: dict):
#     best = None
#     best_area = 0
#     for k, v in resp.items():
#         x1, y1, x2, y2 = v["facial_area"]
#         area = max(0, x2 - x1) * max(0, y2 - y1)
#         if area > best_area:
#             best_area = area
#             best = v
#     return best


# def main():
#     cap = cv2.VideoCapture(0)
#     assert cap.isOpened()

#     dfer = DferClipWrapper(ckpt_path="/home3/tqcj46/DFER-CLIP/checkpoint/DFEW-set1-model.pth", device="cuda")

#      # Single player version: one queue is enough; multiplayer version uses track_id -> deque
#     clip_buf = deque(maxlen=16)

#     frame_idx = 0
#      DETECT_EVERY = 3 # Detect every 3 frames
#     last_face_crop = None
#     last_bbox = None

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break

#         frame_idx += 1

#         if frame_idx % DETECT_EVERY == 0:
#              # RetinaFace supports numpy array input (no need to write files)
#             resp = RetinaFace.detect_faces(frame)  # dict
#             if resp:
#                 face = pick_largest_face(resp)
#                 if face is not None:
#                     x1, y1, x2, y2 = face["facial_area"]
#                     last_bbox = (x1, y1, x2, y2)

#                      # Use extract_faces directly for aligned cropping (align=True)
#                      # Note: The parameter of extract_faces is called img_path, but it can also be passed as numpy array (library code/documentation description)
#                     faces = RetinaFace.extract_faces(img_path=frame, align=True)
#                     if len(faces) > 0:
#                          last_face_crop = faces[0] # This is usually used when single person/largest face

#          # If not detected this round, use the last face crop (simple "keep")
#         if last_face_crop is not None:
#              # Uniform size (DFER-CLIP commonly used 224)
#             face224 = cv2.resize(last_face_crop, (224, 224))
#             clip_buf.append(face224)

#             if len(clip_buf) == 16:
#                 label, score = dfer.predict_clip(list(clip_buf))

#                  #frame + label
#                 if last_bbox is not None:
#                     x1, y1, x2, y2 = last_bbox
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, f"{label} {score:.2f}", (x1, max(0, y1 - 10)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         cv2.imshow("Realtime DFER", frame)
#         if cv2.waitKey(1) & 0xFF == 27:  # ESC
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()






import os
# =======================================================
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# =======================================================
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
# =======================================================

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
# =======================================================
from retinaface import RetinaFace
import argparse


from models.clip import clip
from models.Generate_Model import GenerateModel
from models.Text import *

# =================================================================
# =================================================================
class RecorderMeter(object):
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        pass

    def plot_curve(self, save_path):
        pass
# =================================================================

def main():
    #  ================= 1. Basic configuration =================
    image_path = '/home/wang/projects/facialclip/ku.png'       
    output_path = 'kuku.png'    
    checkpoint_path = '/home/wang/projects/facialclip/checkpoint/DFEW-set1-model.pth' 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    window_size = 16 #  Timing length, DFER-CLIP defaults to 16 frames
    print(f"Currently using device: {device}")

    #  ================= 2. Construct Args to simulate command line parameters =================
    #  Because GenerateModel requires args, fake one for it here
    parser = argparse.ArgumentParser()
    args = parser.parse_args('')
    args.dataset = "DFEW"  #  Assuming using the DFEW dataset
    args.contexts_number = 8
    args.class_token_position = "end"
    args.class_specific_contexts = 'True'
    args.text_type = "class_names"
    args.temporal_layers = 1
    #  👉 Complete the missing parameters required by PromptLearner
    args.load_and_tune_prompt_learner = 'False'
    
    #  Get category label text
    input_text = class_names_7 #  Because it is DFEW, take 7 classification labels
    classes = ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']

    #  ================= 3. Initialize and load the DFER-CLIP model =================
    print("Loading CLIP underlying model...")
    CLIP_model, _ = clip.load("ViT-B/16", device='cpu')
    
    print("Building DFER-CLIP architecture...")
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)
    
    print(f"Loading pre-trained weights: {checkpoint_path} ...")
    #   Weight loading logic in main.py (remove module. prefix)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        pre_trained_dict = checkpoint['state_dict']
    else:
        pre_trained_dict = checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in pre_trained_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()

    #  Standard preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])

    #  ================= 4. Face detection and cropping =================
    print("Read pictures and detect faces...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}, please check the path.")

    faces = RetinaFace.detect_faces(img)
    if not isinstance(faces, dict) or len(faces) == 0:
        print("No face detected!")
        return

    first_face_key = list(faces.keys())[0]
    face_info = faces[first_face_key]
    bbox = faces[first_face_key]['facial_area'] # [x1, y1, x2, y2]
    
    #  === 👇 Added code to obtain key points 👇 ===
    landmarks = face_info['landmarks']
    #  landmarks is a dictionary containing the coordinates of 5 feature points
    #  For example: {'left_eye': [x, y], 'right_eye': [x, y], 'nose': [x, y], 'mouth_left': [x, y], 'mouth_right': [x, y]}
    # ==================================

    x1, y1, x2, y2 = bbox
    h, w, _ = img.shape
    pad = 15
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

    face_crop = img[y1:y2, x1:x2]
    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    
    #  Becomes a Tensor of shape [3, 224, 224]
    face_tensor = transform(face_pil) 

    #  ================= 5. Forge timing and reason =================
    #  Copy a single image into 16 copies, becoming [16, 3, 224, 224]
    video_tensor = face_tensor.unsqueeze(0).repeat(window_size, 1, 1, 1) 
    #  Increase Batch dimension -> [1, 16, 3, 224, 224]
    input_tensor = video_tensor.unsqueeze(0).to(device)

    print("Expression recognition inference in progress...")
    with torch.no_grad():
        output = model(input_tensor)
        
        #  1. Original basic emotion categories
        pred_idx = output.argmax(dim=-1).item()
        emotion_label = classes[pred_idx]

        #  ================= New: Calculate tension =================
        #  2. Convert logits to percentage probabilities
        probabilities = F.softmax(output, dim=-1)[0]
        
        #  3. Define the “tension weight” of the 7 emotions 
        #  Corresponds to: ['Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Fear']
        weights = torch.tensor([-0.8, 0.6, -0.5, 0.2, 0.4, 0.2, 1.0]).to(device)
        
        #  4. Calculate the weighted total score
        raw_score = torch.sum(probabilities * weights).item()
        
        #  5. Map the scores to the range of 0 ~ 100 (assuming extreme relaxation is -0.8 and extreme tension is 1.0)
        nervous_level = ((raw_score - (-0.8)) / (1.0 - (-0.8))) * 100
        nervous_level = max(0, min(100, nervous_level)) #  Limit between 0-100
        
       
        display_text1 = f"Emotion: {emotion_label}"
        display_text2 = f"Tension: {nervous_level:.1f}/100"
        # =======================================================

    print(f"=====================================")
    print(f"Recognition result: {emotion_label}")
    print(f"=====================================")

    #  # ================= 6. Frame and save =================
    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.putText(img, emotion_label, (x1, max(20, y1 - 10)), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    #  # === 👇 Add code for drawing key points 👇 ===
    #  # Traverse 5 key points and draw them with red solid circles
    # for point_name, point_coords in landmarks.items():
    #      # point_coords is a list of floating point numbers, cv2.circle requires an integer tuple
    #     px, py = int(point_coords[0]), int(point_coords[1])
    #     cv2.circle(img, (px, py), radius=3, color=(0, 0, 255), thickness=-1)
    # # ==================================

    # cv2.imwrite(output_path, img)
    #  print(f"Processing completed! The result has been saved to: {output_path}")

    #  ================= 6. Frame and save =================
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    #  === New: Draw key points ===
    #  Traverse 5 key points and draw them with red solid circles
    for point_name, point_coords in landmarks.items():
        #  point_coords is a list of floating point numbers, cv2.circle requires an integer tuple
        px, py = int(point_coords[0]), int(point_coords[1])
        cv2.circle(img, (px, py), radius=3, color=(0, 0, 255), thickness=-1)
    # ====================

    #  Draw the first line: basic emotions (red text)
    cv2.putText(img, display_text1, (x1, max(20, y1 - 35)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
    #  Draw the second line: Tension level (blue text)
    cv2.putText(img, display_text2, (x1, max(45, y1 - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imwrite(output_path, img)
    print(f"Processing completed! Results saved to: {output_path}")
if __name__ == '__main__':
    main()