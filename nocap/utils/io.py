import cv2
import numpy as np
from PIL import Image


def write_video(video_path, frames, fps, width, height):
    assert frames[0].shape[-1] == 3
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (height, width))
    for i, frame in enumerate(frames):
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()