import cv2
import numpy as np
import plotext as plt
from matplotlib import cm

from lookit.cm import FIRE


def colorize(img, cmap="fire", vmin=None, vmax=None):
    """Colorize a single-channel image"""

    assert len(img.shape) < 3 or (len(img.shape) == 3 and img.shape[2] == 1)
    if len(img.shape) == 3:
        img.squeeze(axis=2)

    # Get the min and max
    if vmin is None:
        vmin = img.min()
    if vmax is None:
        vmax = img.max()

    # Clip and rescale
    img = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)

    if cmap is None or cmap == "None":
        return (np.dstack((img, img, img, np.ones((img.shape)))) * 255).astype(np.uint8)

    if cmap == "fire":
        cmap = FIRE
    else:
        cmap = cm.get_cmap(cmap)

    # Apply the colormap
    return (cmap(img) * 255).astype(np.uint8)


def resize_and_pad(img, size, color=0):
    """Take an image of any size and resize it to the target size keeping aspect ratio"""

    h, w = img.shape[:2]
    sh, sw = size[:2]

    # aspect ratio of image
    aspect = w / h
    s_aspect = sw / sh

    # compute scaling and pad sizing
    if aspect > s_aspect:  # Limiting factor is height
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < s_aspect:  # Limiting factor is width
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(
            int
        )
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(
        color, (list, tuple, np.ndarray)
    ):  # color image but only one color provided
        color = [color] * 3

    # interpolation method
    if w > new_w or h > new_h:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # Pad
    if any([p != 0 for p in [pad_top, pad_bot, pad_left, pad_right]]):
        scaled_img = cv2.copyMakeBorder(
            scaled_img,
            pad_top,
            pad_bot,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=color,
        )
    return scaled_img


def summary(img):
    """Display a summary of useful data about an image"""
    img = np.array(img)
    y, x = np.histogram(img, bins=255)

    y = y.astype(float)
    y /= y.sum()
    y *= 100

    plt.colorless()
    plt.bar(x, y)
    plt.xlim(x.min(), x.max())
    plt.xlabel("Pixel Intensity Value")
    plt.ylabel("Percent of Pixels")
    tw, th = plt.terminal_size()
    plt.plotsize(None, th - 2)
    plt.show()

    print(
        "data type: {} | "
        "shape: {} | "
        "height: {} | "
        "width: {} ".format(img.dtype, img.shape, img.shape[0], img.shape[1])
    )


def vcutstack(img, size=None, fill_color=1):
    """Cuts an image in half vertically and stacks it back together horizontally"""
    h = img.shape[0]
    if size is None:
        return np.hstack((img[: int(h / 2), :, :], img[int(h / 2) :, :, :]))

    # Compute how much is left over
    single_h = size[0]
    leftover = int(h / 2) % single_h

    # If there's any left over, add a blank image
    if leftover != 0:
        prev_shape = list(img.shape)
        prev_shape[0] = size[0]
        img = np.vstack(
            (img, np.ones((prev_shape)).astype(img.dtype) * fill_color * 255)
        )

    return vcutstack(img)


def plt2numpy(fig):
    """Convert a matplotlib figure to a numpy array"""
    fig.canvas.draw()
    return np.reshape(
        np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
