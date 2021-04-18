from setuptools import setup


setup(
    name="exppy",
    author="Jacob Pettit",
    url="https://github.com/jfpettit/exppy",
    install_requires=[
        "numpy",
        "stable-baselines3",
        "gym[atari, box2d]",
        "pybullet",
        "matplotlib",
        "click",
        "pandas",
        "altair",
        "optuna"
    ]
)
