import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Sociability_Learning",
    version="1.0",
    packages=["Sociability_Learning"],
    author="Victor Lobato",
    author_email="victor.lobatorios@epfl.ch",
    description="Analysis pipeline for sociability learning experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/Sociability_Learning",
    install_requires=["numpy", "scipy", "scikit-learn", "scikit-image", "scikit_posthocs", "opencv-python", "pandas", "tqdm","PyYAML","utils2p"]
)
