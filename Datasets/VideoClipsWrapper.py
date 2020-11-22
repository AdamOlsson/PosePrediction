# torch
from torchvision.datasets.video_utils import VideoClips
from torch.utils.data import Dataset

# native
from os.path import join, dirname

class VideoClipsWrapper(Dataset):
    def __init__(self, video_names, clip_length_in_frames, frames_between_clips, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.video_clips = VideoClips(self.video_names, clip_length_in_frames=clip_length_in_frames, frames_between_clips=frames_between_clips)

    def __len__(self):
        return len(self.video_clips)

    def __getitem__(self, idx):
        vframes, _, info, video_idx = self.video_clips.get_clip(idx)
        sample = {"data":vframes.numpy(), "info":info, "video_idx":video_idx, "type":"video"}
        if self.transform:
            sample = self.transform(sample)

        return sample