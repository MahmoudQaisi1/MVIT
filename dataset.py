import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {j: i for i, j in enumerate(self.classes)}
        self.videos = []
        self.labels = []
        self.current_video = None
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for root, dirs, _ in os.walk(class_path):
                for vid in dirs:
                    frames = sorted(os.listdir(os.path.join(root, vid)))
                    for i in range(len(frames) - 16):
                        self.videos.append((root, vid, i))
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        path, video, start_frame = self.videos[idx]
        frames = []
        for frame in sorted(os.listdir(os.path.join(path, video)))[start_frame:start_frame+16]:
            image = Image.open(os.path.join(path, video, frame))
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        frames = torch.stack(frames)
        label = self.labels[idx]
        class_name = self.classes[label]

        if self.current_video != video:
            self.current_video = video
            print(f'Starting new video: {video} - {class_name}')
        return frames, label
