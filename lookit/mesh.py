import os
import logging

import pyrender
import trimesh
import numpy as np
from PIL import Image


try:
    os.environ["PYOPENGL_PLATFORM"]
except KeyError:
    logging.debug("Setting $PYOPENGL_PLATFORM to egl")
    os.environ["PYOPENGL_PLATFORM"] = "egl"


def create_gif_rot(
    models,
    zoom=2.0,
    num_renders=24,
    start_angle=0,
    **kwargs,
):
    """Create a gif of an object rotating in place"""
    img_list = []
    for angle in range(0, 360, int(360 / num_renders)):
        img_list.append(
            np.expand_dims(
                render(models, yrot=start_angle + angle, ztrans=zoom, **kwargs),
                axis=3,
            )
        )
    return np.concatenate(img_list, axis=3)


def create_gif_tf(
    models,
    transforms,
    zoom=2.0,
    **kwargs,
):
    """
    Create a gif using a list of custom transforms

    models should be a list, transforms should be a list of lists.
    """

    if not isinstance(models, list):
        models = [models]

    img_list = []
    for tf in transforms:
        img_list.append(
            np.expand_dims(
                render(
                    [m.apply_transform(t) for m, t in zip(models, tf)],
                    ztrans=zoom,
                    **kwargs,
                ),
                axis=3,
            )
        )
    return np.concatenate(img_list, axis=3)


def trimesh_simplify(mesh):
    """Convert a trimesh mesh or scene into a trimesh mesh."""

    if isinstance(mesh, list):
        return [trimesh_simplify(m) for m in mesh]

    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            mesh = trimesh.Trimesh()
        else:
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    if hasattr(g, "visual")
                    else trimesh.Trimesh(
                        vertices=g.vertices, faces=g.faces, visual=g.visual
                    )
                    for g in mesh.geometry.values()
                )
            )

    return mesh


def trimesh_pretty_pointcloud(pointcloud, radius=None, subdivisions=2):
    """Create a pointcloud with spheres instead of points"""
    if radius is None:
        radius = max(pointcloud.extents) / 35
    return trimesh.util.concatenate(
        [
            trimesh.primitives.Sphere(
                radius=radius,
                center=v,
                subdivisions=subdivisions,
            )
            for v in pointcloud.vertices
        ]
    )


def trimesh_normalize_matrix(mesh, scale=True):
    """Obtain normalization matrix for mesh so that it occupies a unit cube"""

    # Get the overall size of the object
    mesh_min, mesh_max = np.min(mesh.vertices, axis=0), np.max(mesh.vertices, axis=0)
    size = mesh_max - mesh_min

    # Center the object
    mesh.vertices = (size / 2.0) + mesh_min
    trans = trimesh.transformations.translation_matrix(-((size / 2.0) + mesh_min))
    if not scale:
        return trans

    # Normalize scale of the object
    scale = trimesh.transformations.scale_matrix((1.0 / np.max(size)))
    return scale @ trans


def trimesh_normalize(mesh, scale=True):
    """Normalize a mesh so that it occupies a unit cube"""

    mat = trimesh_normalize_matrix(mesh, scale=scale)
    mesh = mesh.copy()
    mesh.apply_transform(mat)
    return mesh


def render(
    mesh,
    modality="color",
    yfov=(np.pi / 4.0),
    resolution=(1280, 720),
    xtrans=0.0,
    ytrans=0.0,
    ztrans=2.0,
    xrot=-25.0,
    yrot=45.0,
    zrot=0.0,
    spotlight_intensity=8.0,
    remove_texture=False,
    wireframe=False,
    bg_color=255,
    mode="RGB",
    point_size=5,
):
    """Render a trimesh object or list of objects"""

    assert modality in ["color", "depth", "pointcloud"]

    # Create a pyrender scene with ambient light
    scene = pyrender.Scene(ambient_light=np.ones(3), bg_color=bg_color)

    mesh = trimesh_simplify(mesh)
    if not isinstance(mesh, list):
        mesh = [mesh]
    if remove_texture:
        for i in range(len(mesh)):
            if isinstance(mesh[i], trimesh.Trimesh):
                mesh[i] = trimesh.Trimesh(
                    vertices=mesh[i].vertices, faces=mesh[i].faces
                )
    if not isinstance(wireframe, list):
        wireframe = [wireframe] * len(mesh)
    for m, w in zip(mesh, wireframe):
        if isinstance(m, trimesh.points.PointCloud):
            scene.add(pyrender.Mesh.from_points(m.vertices, colors=m.colors))
        else:
            scene.add(pyrender.Mesh.from_trimesh(m, wireframe=w))

    aspect_ratio = resolution[0] / resolution[1]
    camera = pyrender.PerspectiveCamera(
        yfov=yfov,
        aspectRatio=aspect_ratio,
    )

    # Apply translations
    trans = np.array(
        [
            [1.0, 0.0, 0.0, xtrans],
            [0.0, 1.0, 0.0, ytrans],
            [0.0, 0.0, 1.0, ztrans],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Apply rotations
    xrotmat = trimesh.transformations.rotation_matrix(
        angle=np.radians(xrot), direction=[1, 0, 0], point=(0, 0, 0)
    )
    camera_pose = np.dot(xrotmat, trans)
    yrotmat = trimesh.transformations.rotation_matrix(
        angle=np.radians(yrot), direction=[0, 1, 0], point=(0, 0, 0)
    )
    camera_pose = np.dot(yrotmat, camera_pose)
    zrotmat = trimesh.transformations.rotation_matrix(
        angle=np.radians(zrot), direction=[0, 0, 1], point=(0, 0, 0)
    )
    camera_pose = np.dot(zrotmat, camera_pose)

    # Insert the camera
    scene.add(camera, pose=camera_pose)

    # Insert a splotlight to give contrast
    spot_light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=spotlight_intensity,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    scene.add(spot_light, pose=camera_pose)

    # Render!
    r = pyrender.OffscreenRenderer(resolution[0], resolution[1], point_size=point_size)
    c, d = r.render(scene)
    if modality == "depth":
        return np.array(Image.fromarray(d).convert(mode))
    elif modality == "color":
        return np.array(Image.fromarray(c).convert(mode))

    def pointcloud(depth, fx, fy, camera_matrix):
        # fy = fx = 0.5 / np.tan(fov * 0.5)
        height = depth.shape[0]
        width = depth.shape[1]
        mask = np.where(depth > 0)
        x = mask[1]
        y = mask[0]
        normalized_x = (x.astype(float) - width * 0.5) / width
        normalized_y = -(y.astype(float) - height * 0.5) / height
        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        world_z = -depth[y, x]
        pointcloud = trimesh.points.PointCloud(np.vstack((world_x, world_y, world_z)).T)
        pointcloud.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(180), (0, 0, 1))
            @ trimesh.transformations.rotation_matrix(np.radians(180), (0, 1, 0))
            @ np.linalg.inv(camera_matrix)
        )
        return pointcloud

    return pointcloud(
        depth=np.array(Image.fromarray(d).convert("L")),
        fx=(yfov * aspect_ratio),
        fy=yfov,
        camera_matrix=camera_pose,
    )
