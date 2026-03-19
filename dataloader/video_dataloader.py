# import os.path
# from numpy.random import randint
# from torch.utils import data
# import glob
# import os
# from dataloader.video_transform import *
# import numpy as np


# class VideoRecord(object):
#     def __init__(self, row):
#         self._data = row

#     @property
#     def path(self):
#         return self._data[0]

#     @property
#     def num_frames(self):
#         return int(self._data[1])

#     @property
#     def label(self):
#         return int(self._data[2])


# class VideoDataset(data.Dataset):
#     def __init__(self, list_file, num_segments, duration, mode, transform, image_size):

#         self.list_file = list_file
#         self.duration = duration
#         self.num_segments = num_segments
#         self.transform = transform
#         self.image_size = image_size
#         self.mode = mode
#         self._parse_list()
#         pass

#     def _parse_list(self):
#         #
#         # Data Form: [video_id, num_frames, class_idx]
#         #
#         tmp = [x.strip().split(' ') for x in open(self.list_file)]
#         tmp = [item for item in tmp]
#         self.video_list = [VideoRecord(item) for item in tmp]
#         print(('video number:%d' % (len(self.video_list))))

#     def _get_train_indices(self, record):
#         # 
#         # Split all frames into seg parts, then select frame in each part randomly
#         #
#         average_duration = (record.num_frames - self.duration + 1) // self.num_segments
#         if average_duration > 0:
#             offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
#         elif record.num_frames > self.num_segments:
#             offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
#         else:
#             offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
#         return offsets

#     def _get_test_indices(self, record):
#         # 
#         # Split all frames into seg parts, then select frame in the mid of each part
#         #
#         if record.num_frames > self.num_segments + self.duration - 1:
#             tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
#             offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
#         else:
#             offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
#         return offsets

#     def __getitem__(self, index):
#         record = self.video_list[index]
#         if self.mode == 'train':
#             segment_indices = self._get_train_indices(record)
#         elif self.mode == 'test':
#             segment_indices = self._get_test_indices(record)
        
#         return self.get(record, segment_indices)

#     def get(self, record, indices):
#         video_frames_path = glob.glob(os.path.join(record.path, '*'))
#         video_frames_path.sort()
#         images = list()
#         for seg_ind in indices:
#             p = int(seg_ind)
#             for i in range(self.duration):
#                 seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert('RGB')]
#                 images.extend(seg_imgs)
#                 if p < record.num_frames - 1:
#                     p += 1

#         images = self.transform(images)
#         images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

#         return images, record.label

#     def __len__(self):
#         return len(self.video_list)


# def train_data_loader(list_file, num_segments, duration, image_size, args):
    
#     if args.dataset == "DFEW":
#         train_transforms = torchvision.transforms.Compose([
#             ColorJitter(brightness=0.5),
#             GroupRandomSizedCrop(image_size),
#             GroupRandomHorizontalFlip(),
#             Stack(),
#             ToTorchFormatTensor()])
#     elif args.dataset == "FERV39K":
#         train_transforms = torchvision.transforms.Compose([
#             RandomRotation(4),
#             GroupRandomSizedCrop(image_size),
#             GroupRandomHorizontalFlip(),
#             Stack(),
#             ToTorchFormatTensor()])  
#     elif args.dataset == "MAFW":
#         train_transforms = torchvision.transforms.Compose([
#             GroupRandomSizedCrop(image_size),
#             GroupRandomHorizontalFlip(),
#             Stack(),
#             ToTorchFormatTensor()]) 
    
#     # train_transforms = torchvision.transforms.Compose([
#     #         GroupRandomSizedCrop(image_size),
#     #         GroupRandomHorizontalFlip(),
#     #         Stack(),
#     #         ToTorchFormatTensor()])
    
#     train_data = VideoDataset(list_file=list_file,
#                               num_segments=num_segments,
#                               duration=duration,
#                               mode='train',
#                               transform=train_transforms,
#                               image_size=image_size)
#     return train_data


# def test_data_loader(list_file, num_segments, duration, image_size):
    
#     test_transform = torchvision.transforms.Compose([GroupResize(image_size),
#                                                      Stack(),
#                                                      ToTorchFormatTensor()])
    
#     test_data = VideoDataset(list_file=list_file,
#                              num_segments=num_segments,
#                              duration=duration,
#                              mode='test',
#                              transform=test_transform,
#                              image_size=image_size)
#     return test_data
import os.path
from numpy.random import randint
from torch.utils import data
import glob
import os
from dataloader.video_transform import *
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
import torchvision
import re

#  Allowed image extensions
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def is_image_file(p):
    return os.path.isfile(p) and (os.path.splitext(p)[1].lower() in IMG_EXTS)

def natural_key(s):
    #  Natural number sorting: 1,2,10 instead of 1,10,2
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_image_paths_sorted(folder):
    paths = [p for p in glob.glob(os.path.join(folder, '*')) if is_image_file(p)]
    paths.sort(key=natural_key)
    return paths


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size):

        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self._parse_list()

    def _parse_list(self):
        #
        # Data Form: [video_id, num_frames, class_idx]
        #
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp]
        self.video_list = [VideoRecord(item) for item in tmp]
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record, n_frames=None):
        frames = record.num_frames if n_frames is None else n_frames
        average_duration = (frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                      randint(average_duration, size=self.num_segments)
        elif frames > self.num_segments:
            offsets = np.sort(randint(frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(frames))),
                             (0, self.num_segments - frames), 'edge')
        return offsets

    def _get_test_indices(self, record, n_frames=None):
        frames = record.num_frames if n_frames is None else n_frames
        if frames > self.num_segments + self.duration - 1:
            tick = (frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(frames))),
                             (0, self.num_segments - frames), 'edge')
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        #  Only take legal pictures and sort them naturally
        video_frames_path = list_image_paths_sorted(record.path)
        n = len(video_frames_path)
        if n == 0:
            raise FileNotFoundError(f"No valid image frames found in: {record.path}")

        if n != record.num_frames:
            print(f"[Warn] Frame count mismatch for {record.path}: "
                  f"listed={record.num_frames}, actual={n}")

        if self.mode == 'train':
            segment_indices = self._get_train_indices(record, n_frames=n)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record, n_frames=n)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return self.get(record, segment_indices, video_frames_path)

    def _safe_open_rgb(self, path):
        try:
            return Image.open(path).convert('RGB')
        except (UnidentifiedImageError, OSError):
            return None

    def get(self, record, indices, video_frames_path):
        n = len(video_frames_path)
        images = []
        last_valid_img = None

        for seg_ind in indices:
            p = int(seg_ind)
            if p >= n:  #  Index out-of-bounds protection
                p = n - 1

            for _ in range(self.duration):
                img = self._safe_open_rgb(video_frames_path[p])
                if img is None:
                    #  If the current frame is corrupted, try to find a nearby readable frame
                    #  Look first and backward
                    q = p + 1
                    while q < n:
                        img = self._safe_open_rgb(video_frames_path[q])
                        if img is not None:
                            break
                        q += 1
                    #  Look forward again
                    if img is None:
                        q = p - 1
                        while q >= 0:
                            img = self._safe_open_rgb(video_frames_path[q])
                            if img is not None:
                                break
                            q -= 1
                    #  If still not found, fall back to the previous valid frame (if any)
                    if img is None and last_valid_img is not None:
                        img = last_valid_img.copy()

                    #  If it is still None in the end, an error will be reported (indicating that the entire section is broken)
                    if img is None:
                        raise FileNotFoundError(
                            f"All nearby frames unreadable around index {p} in {os.path.dirname(video_frames_path[p])}"
                        )

                images.append(img)
                last_valid_img = img

                if p < n - 1:
                    p += 1

        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))
        return images, record.label

    def __len__(self):
        return len(self.video_list)


def train_data_loader(list_file, num_segments, duration, image_size, args):

    if args.dataset == "DFEW":
        train_transforms = torchvision.transforms.Compose([
            ColorJitter(brightness=0.5),
            GroupRandomSizedCrop(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
    elif args.dataset == "FERV39K":
        train_transforms = torchvision.transforms.Compose([
            RandomRotation(4),
            GroupRandomSizedCrop(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
    elif args.dataset == "MAFW":
        train_transforms = torchvision.transforms.Compose([
            GroupRandomSizedCrop(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
    else:
        train_transforms = torchvision.transforms.Compose([
            GroupRandomSizedCrop(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])

    train_data = VideoDataset(list_file=list_file,
                              num_segments=num_segments,
                              duration=duration,
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size)
    return train_data


def test_data_loader(list_file, num_segments, duration, image_size):

    test_transform = torchvision.transforms.Compose([
        GroupResize(image_size),
        Stack(),
        ToTorchFormatTensor()])

    test_data = VideoDataset(list_file=list_file,
                             num_segments=num_segments,
                             duration=duration,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size)
    return test_data
