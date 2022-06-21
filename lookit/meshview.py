import os
import argparse

import tqdm
import cv2
import trimesh
import numpy as np
from PIL import Image

import lookit
from lookit import LOG


def render_ext(args):
    ext = args.ext
    if ext[0] != ".":
        ext = "." + ext

    root_dir = os.path.abspath(args.root)
    assert os.path.isdir(root_dir), "Root dir not found: {}".format(args.splits)
    assert len(args.resolution) == 2
    save_file = os.path.abspath(args.save)
    assert os.path.isdir(
        os.path.dirname(save_file)
    ), "Parent dir does not exist: {}".format(save_file)

    resolution = args.resolution
    num_renders = args.num_renders
    angle_list = range(0, 360, int(360 / num_renders))

    # Render
    stacker = []
    f_list = []
    idx = 0
    try:
        pbar = tqdm.tqdm(lookit.utils.get_file(root_dir, fe=ext, fn=args.name))

        for f in pbar:
            h_stacker = []

            f_list.append(f)
            idx += 1

            try:
                mesh = lookit.mesh.trimesh_normalize(trimesh.load(f))
                for a in angle_list:
                    img = lookit.mesh.render(
                        mesh=mesh,
                        resolution=resolution[:2],
                        mode="RGB",
                        yrot=a,
                        spotlight_intensity=8.0,
                        remove_texture=args.remove_texture,
                        bg_color=0,
                    )
                    img = cv2.putText(
                        img, str(idx), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1
                    )
                    h_stacker.append(img)
            except (ValueError, AttributeError):
                pbar.write("Error loading mesh {}".format(f))
                continue

            stacker.append(np.hstack(h_stacker))
    except KeyboardInterrupt:
        pass

    LOG.info("File list:")
    for idx, f in enumerate(f_list):
        LOG.info("{}: {}".format(idx, f))
    if args.save_paths:
        with open(args.save_paths, "w") as file:
            for idx, f in enumerate(f_list):
                file.write("{}:{}\n".format(idx, f))

    # Create one big image
    stacker = np.vstack(stacker)
    while (stacker.shape[0] > stacker.shape[1]) and (stacker.shape[1] < 2**31):
        stacker = lookit.image.vcutstack(stacker, resolution)

    # Save
    Image.fromarray(stacker).save(save_file)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Create a render for an entire dataset and save it to disk."
    )
    arg_parser.add_argument(
        "root",
        type=str,
        help="Path to the root directory.",
    )
    arg_parser.add_argument(
        "save",
        type=str,
        help="Path to save the render at.",
    )
    arg_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of files to render.",
    )
    arg_parser.add_argument(
        "--ext",
        default=".obj",
        help="If passed, use this extension.",
    )
    arg_parser.add_argument(
        "--save_paths",
        type=str,
        default=None,
        help="If passed, will save the list of filenames to a file.",
    )
    arg_parser.add_argument(
        "--num_renders",
        type=int,
        default=1,
        help="Number of renders to generate.",
    )
    arg_parser.add_argument(
        "--remove_texture",
        action="store_true",
        default=False,
        help="If passed, will strip the texture out of an object.",
    )
    arg_parser.add_argument(
        "--resolution",
        nargs=2,
        default=[200, 200],
        type=int,
        help="Resolution to render at.",
    )
    args = arg_parser.parse_args()

    render_ext(args)
