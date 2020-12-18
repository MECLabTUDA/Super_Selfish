import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="super-selfish",
    version="0.0.6",
    author="Nicolas Wagner",
    author_email="nicolas_wagner@gmx.de",
    description="A self-supervision PyTorch framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nwWag/Super-Selfish",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
