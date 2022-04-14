import imageio
import numpy as np
from PIL import Image


def load_gif(
    f_in,
):
    """Load a gif"""
    gif = imageio.get_reader(f_in)
    frames = []
    for idx, f in enumerate(gif):
        frames.append(np.array(f))
    if len(frames[0].shape) == 2:
        frames = [np.expand_dims(f, axis=2) for f in frames]
    frames = [np.expand_dims(f, axis=3) for f in frames]
    return np.concatenate(frames, axis=3)


def save_gif(
    f_out,
    data,
    **kwargs,
):
    """Save a list of images as a gif"""
    data = [Image.fromarray(data[:, :, :, i]) for i in range(data.shape[3])]
    data[0].save(f_out, save_all=True, append_images=data[1:], **kwargs)