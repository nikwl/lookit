# lookit
A toolbox of eccentric functions for visualizing and processing images and meshes.

## Installation
To install:
```bash
pip install git+https://github.com/nikwl/lookit.git
```

## Functions

Load a mesh, render, and view a mesh.
```python
import lookit

from PIL import Image
import trimesh

Image.fromarray(
    lookit.mesh.render(
        mesh=trimesh.load("examples/mesh.ply"),
        resolution=(1920, 1080),
        mode="RGB",
    )
).show()
```

Create a point cloud from a given camera perspective, then render and view it.
```python
import lookit

from PIL import Image
import trimesh

ptcld = lookit.mesh.render(
    mesh=trimesh.load("examples/mesh.ply"),
    resolution=(1920, 1080),
    modality="pointcloud",
)

Image.fromarray(
    lookit.mesh.render(
        mesh=lookit.mesh.trimesh_normalize(
            trimesh.PointCloud(ptcld)
        ),
        resolution=(1920, 1080),
        yrot=0,
        xrot=180,
    )
).show()
```
