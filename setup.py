from setuptools import setup, find_packages

setup(
    name="auradsp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "librosa",
        "scipy",
        "numpy",
        "soundfile",
        "rich",
        "click",
        "scikit-learn",
        "pydub",
        "mutagen",
    ],
    entry_points={
        "console_scripts": [
            "auradsp=src.cli:main",
        ],
    },
)
