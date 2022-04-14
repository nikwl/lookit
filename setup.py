import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="lookit",
    version="0.0.1",
    author="Nikolas lamb",
    author_email="nikolas.lamb@gmail.com",
    description="Toolbox for viewing images and meshes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nikwl/lookit",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
		"TOPIC :: MULTIMEDIA :: GRAPHICS :: 3D RENDERING"
    ],
    python_requires='>=3.6',
)