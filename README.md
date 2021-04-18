# exppy

This repository holds code for experiments and code associated with blog posts that I write. Check out my blog [here](jacobpettit.com).

Files:
1. [`optuna_rl.py`](https://github.com/jfpettit/exppy/blob/main/optuna_rl.py) Code from the blog post [Weird RL with Hyperparameter Optimizers](https://www.jacobpettit.com/weird-rl-with-hyperparameter-optimizers/). Part 2 of the [blog post](https://www.jacobpettit.com/weird-rl-part-2-training-in-the-browser/) introduces a [Streamlit webapp](https://intense-savannah-69104.herokuapp.com).
2. [`atari_mask.py`](https://github.com/jfpettit/exppy/blob/main/atari_mask.py) Code from the blog post [Breaking a Pong-playing RL agent](https://www.jacobpettit.com/breaking-a-pong-playing-rl-agent/).


## Setup

Clone the repository and install it:
```bash
git clone https://github.com/jfpettit/exppy.git
cd exppy
pip install -e .
```

If you don't want to install it as a package, then you don't need to, and you can run each file individually using `python file_name.py --args`, provided that you've got all of the required packages installed.

## Running

If you have opted to install it, then you can run things by invoking `python -m`:

```bash
python -m exppy.file_name
```

For example, print out the help message from the Optuna RL code:

```bash
python -m exppy.optuna_rl --help
```

### Usage in files

You can import from the `exppy` module like a normal python package:

```python
from exppy.atari_mask import ImageObsMask
from exppy.optuna_rl import Runner, run_policy, video_rollout
import gym

env = gym.make("BreakoutNoFrameskip-v4")
env = ImageObsMask(env)
```

