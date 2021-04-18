import matplotlib.pyplot as plt
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from gym.wrappers import Monitor
import os
import click


class ImageObsMask(gym.Wrapper):
    def __init__(self, env, fill_val=0, width=25, height=25, xloc=75, yloc=75):
        super().__init__(env)

        self.width = width
        self.height = height
        self.xloc = xloc
        self.yloc = yloc
        self.fill_val = fill_val

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs[self.yloc-self.height//2:self.yloc+self.height//2, self.xloc-self.width//2:self.xloc+self.width//2] = self.fill_val
        return obs, rew, done, info

    def render(self, mode='human'):
        img = self.env.ale.getScreenRGB2()
        img[self.yloc-self.height//2:self.yloc+self.height//2, self.xloc-self.width//2:self.xloc+self.width//2] = self.fill_val
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

def build_env(fill_val, width, height, xloc, yloc):
    env = gym.make("PongNoFrameskip-v4")
    env = ImageObsMask(env, fill_val, width, height, xloc, yloc)
    env_fn = lambda: Monitor(env, f"atari_vids/", force=True)
    env = make_atari_env(env_fn, n_envs=1)
    env = VecFrameStack(env, n_stack=4)
    return env

@click.command()
@click.option("--episodes", "-eps", type=int, default=10, help="Number of episodes to evaluate over.")
@click.option("--fill-val", "-fill", type=float, default=0., help="Pixel value to fill mask with.")
@click.option("--width", "-w", type=int, default=25, help="Width of mask.")
@click.option("--height", "-h", type=int, default=25, help="Height of mask.")
@click.option("--xloc", "-x", type=int, default=75, help="X location to center mask on in image.")
@click.option("--yloc", "-y", type=int, default=75, help="Y location to center mask on in image.")
def run(episodes, fill_val, width, height, xloc, yloc):

    env = build_env(fill_val, width, height, xloc, yloc)
    model = PPO.load("models/PongNoFrameskip-v4_1618178168.zip", env=env)
    mean_rew, std_rew = evaluate_policy(model, model.get_env(), n_eval_episodes=episodes, deterministic=True)

    path = os.getcwd() + "/atari_vids/"
    print(f"Videos saved to: {path}")

if __name__ == "__main__":
    run()
