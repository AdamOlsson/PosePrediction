from paf.util import load_humans
from torch import from_numpy
import numpy as np
import torchvision, cv2
from paf.common import draw_humans

def write_video(path, video_array):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(path, fourcc, 18, (video_array.shape[1], video_array.shape[2]))

    for frame in video_array[:]:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

path = "data/humans_video.json"
data = load_humans(path)
metadata, frames = data["metadata"], data["frames"]

vframes, _, _ = torchvision.io.read_video(metadata["filename"], pts_unit="sec") # Tensor[T, H, W, C]) – the T video frames
vframes = np.flip(vframes.numpy(), axis=3)

no_frames = vframes.shape[0]
selected_frames = np.linspace(0, no_frames-1, num=int(no_frames/metadata["frame_skip"]), dtype=np.int)
vframes = vframes[selected_frames]

for frame_idx in range(len(vframes)):
    vframes[frame_idx] = draw_humans(vframes[frame_idx], frames[frame_idx])

vframes = np.flip(vframes, axis=3).copy()
torchvision.io.write_video("docs/result.mp4", from_numpy(vframes), 12)