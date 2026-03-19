import argparse
import torch
import os
import numpy as np

#  1. Must be set before all torch.load: allow loading of custom classes
from main import RecorderMeter 
torch.serialization.add_safe_globals([RecorderMeter])

import torch.nn as nn
from models.Generate_Model import GenerateModel
from clip import clip
from dataloader.video_dataloader import test_data_loader
from models.Text import *
from main import computer_uar_war 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DFEW')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--exper-name', type=str, required=True, help="Timestamp during training + experiment name")
    parser.add_argument('--contexts-number', type=int, default=8)
    parser.add_argument('--class-token-position', type=str, default="end")
    parser.add_argument('--class-specific-contexts', type=str, default='True')
    parser.add_argument('--load-and-tune-prompt-learner', type=str, default='False')
    parser.add_argument('--text-type', type=str, default='class_descriptor')
    parser.add_argument('--temporal-layers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

def test():
    args = get_args()
    #  Force the current primary card to be specified
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #  1. First load the CLIP and make sure it is on the GPU. This is the core of solving RuntimeError.
    #  Only if CLIP is on the GPU, the internal tensors will be allocated correctly when GenerateModel is initialized later.
    CLIP_model, _ = clip.load("ViT-B/32", device=device) 
    CLIP_model = CLIP_model.float() #  <--- add this line
    CLIP_model.eval()

    if args.dataset == "DFEW":
        all_fold = 5
        input_text = {
            "class_names": class_names_7,
            "class_names_with_context": class_names_with_context_7,
            "class_descriptor": class_descriptor_7
        }.get(args.text_type, class_descriptor_7)
    
    UAR, WAR = 0.0, 0.0

    #  Loop evaluates 5 Folds
    for data_set in range(1, all_fold + 1):
        print(f"\n>>>> Testing Fold {data_set} <<<<")
        
        #  2. Initialize the model. At this point CLIP is already on the GPU and PromptLearner's calculations will remain consistent.
        model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)
        model = model.to(device).float() #  <--- Modify this line

        checkpoint_name = f"{args.dataset}-{args.exper_name}-set{data_set}-model_best.pth"
        best_checkpoint_path = f"./checkpoint/{checkpoint_name}"
        
        if not os.path.exists(best_checkpoint_path):
            print(f"Warning: Weight file {best_checkpoint_path} not found, skipping this Fold.")
            continue

        log_txt_path = f"./log/TEST-{checkpoint_name}.txt"
        log_confusion_matrix_path = f"./log/TEST-{checkpoint_name}-cm.png"
        test_annotation_file_path = f"./annotation/{args.dataset}_set_{data_set}_test.txt"

        #  3. Load test data
        test_data = test_data_loader(list_file=test_annotation_file_path, 
                                     num_segments=16, duration=1, image_size=224)
        val_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, 
                                                 shuffle=False, num_workers=args.workers, pin_memory=True)

        #  4. Package parallelization. Note: Be sure to ensure that the loading logic can handle the DataParallel prefix.
        #  Here we wrap it first, because computer_uar_war will try to load and process it internally again.
        model = torch.nn.DataParallel(model)

        try:
            uar, war = computer_uar_war(val_loader, model, best_checkpoint_path, 
                                        log_confusion_matrix_path, log_txt_path, data_set)
            UAR += float(uar)
            WAR += float(war)
            print(f"Fold {data_set} Result: UAR={uar:.2f}, WAR={war:.2f}")
        except Exception as e:
            print(f"Error evaluating Fold {data_set}: {e}")

    print('\n' + '='*40)
    print(f'Final Average Results ({all_fold} Folds):')
    print(f"Overall Mean UAR: {UAR/all_fold:.2f}")
    print(f"Overall Mean WAR: {WAR/all_fold:.2f}")
    print('='*40)

if __name__ == '__main__':
    test()