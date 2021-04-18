import numpy as np
import pandas as pd
import gym
import pybullet_envs
import altair as alt

import optuna
import click
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    x[x < 0] = 0
    return x

def linear(x):
    return x

class Policy:
    def __init__(self):
        pass

    def __call__(self):
        pass


class DensePolicy(Policy):
    def __init__(self, kernel=None, env=None, activation=linear):
        assert isinstance(env.action_space, gym.spaces.Box), "DenseGaussianPolicy requires Box action spaces."
        self.kernel = kernel
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.activation = activation

    def __call__(self, state):
        out = np.dot(self.kernel, state)
        act = self.activation(out)
        return act

    def load(self, params, env):
        params_k = np.array([v for k, v in params.items() if "kernel" in k]).reshape((env.action_space.shape[0], env.observation_space.shape[0]))
        self.__init__(params_k, env)


class DenseGaussianPolicy(Policy):
    def __init__(self, kernel=None, vars_=None, env=None, activation=linear):
        assert isinstance(env.action_space, gym.spaces.Box), "DenseGaussianPolicy requires Box action spaces."
        self.kernel = kernel
        self.vs = vars_
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.activation = activation

    def __call__(self, state):
        mus = np.dot(self.kernel, state)
        mus = self.activation(mus)
        act = mus + np.random.randn(self.action_dim) * self.vs
        return act

    def load(self, params, env):
        params_k = np.array([v for k, v in params.items() if "kernel" in k]).reshape((env.action_space.shape[0], env.observation_space.shape[0]))
        params_v = np.array([v for k, v in params.items() if "vars" in k])

        self.__init__(params_k, params_v, env)


class SigmoidDensePolicy(Policy):
    def __init__(self, kernel=None, env=None, activation=sigmoid):
        self.pol = DensePolicy(kernel, env, activation=activation)

    def __call__(self, state):
        return self.pol(state)

    def load(self, params, env):
        self.pol.load(params, env)


class SigmoidDenseGaussianPolicy(Policy):
    def __init__(self, kernel=None, vars_=None, env=None, activation=sigmoid):
        self.pol = DenseGaussianPolicy(kernel, vars_=vars_, env=env, activation=activation)

    def __call__(self, state):
        return self.pol(state)

    def load(self, params, env):
        self.pol.load(params, env)

def optimize_gdense_policy(trial, env_name, n_episodes=5, T=10000, min_param=-3., max_param=3., var_max=3.):
    env = gym.make(env_name)

    kernel = np.zeros((env.action_space.shape[0], env.observation_space.shape[0]))
    vs = np.zeros(env.action_space.shape[0])
    ctr = 0

    for i in range(env.action_space.shape[0]):
        vs[i] = trial.suggest_float("vars"+str(i), 0, var_max)
        for j in range(env.observation_space.shape[0]):
            kernel[i,j] = trial.suggest_float("kernel"+str(ctr), min_param, max_param)
            ctr += 1

    policy = DenseGaussianPolicy(kernel, vs, env)
    result = run_policy(policy, env_name, n_episodes=n_episodes, T=T)

    return result

def optimize_dense_policy(trial, env_name, n_episodes=5, T=10000, min_param=-3., max_param=3., var_max=3.):
    env = gym.make(env_name)

    kernel = np.zeros((env.action_space.shape[0], env.observation_space.shape[0]))
    ctr = 0

    for i in range(env.action_space.shape[0]):
        for j in range(env.observation_space.shape[0]):
            kernel[i,j] = trial.suggest_float("kernel"+str(ctr), min_param, max_param)
            ctr += 1

    policy = DensePolicy(kernel, env)
    result = run_policy(policy, env_name, n_episodes=n_episodes, T=T)

    return result

def optimize_sdense_policy(trial, env_name, n_episodes=5, T=10000, min_param=-3., max_param=3., var_max=3.):
    env = gym.make(env_name)

    kernel = np.zeros((env.action_space.shape[0], env.observation_space.shape[0]))
    ctr = 0

    for i in range(env.action_space.shape[0]):
        for j in range(env.observation_space.shape[0]):
            kernel[i,j] = trial.suggest_float("kernel"+str(ctr), min_param, max_param)
            ctr += 1

    policy = SigmoidDensePolicy(kernel, env)
    result = run_policy(policy, env_name, n_episodes=n_episodes, T=T)

    return result

def optimize_sgdense_policy(trial, env_name, n_episodes=5, T=10000, min_param=-3., max_param=3., var_max=3.):
    env = gym.make(env_name)

    kernel = np.zeros((env.action_space.shape[0], env.observation_space.shape[0]))
    vs = np.zeros(env.action_space.shape[0])
    ctr = 0

    for i in range(env.action_space.shape[0]):
        vs[i] = trial.suggest_float("vars"+str(i), 0, var_max)
        for j in range(env.observation_space.shape[0]):
            kernel[i,j] = trial.suggest_float("kernel"+str(ctr), min_param, max_param)
            ctr += 1

    policy = SigmoidDenseGaussianPolicy(kernel, vs, env)
    result = run_policy(policy, env_name, n_episodes=n_episodes, T=T)

    return result

def run_policy(policy, env_name, n_episodes=5, T=1000):
    env = gym.make(env_name)

    Rs = []
    lens = []
    for n in range(n_episodes):
        obs = env.reset()
        R = 0
        l = 0

        for t in range(T):
            action = policy(obs)
            obs, rew, done, infos = env.step(action)

            R += rew
            l += 1

            if done:
                Rs.append(R)
                lens.append(l)
                R = 0
                l = 0
                break

    return np.mean(Rs)

def video_rollout(policy, params, env_name, n_episodes, horizon, save_dir):
    env = gym.make(env_name)
    env = gym.wrappers.Monitor(env, directory=save_dir, force=True)

    policy = policy(env=env)
    policy.load(params, env)

    Rs = []
    lens = []
    for n in range(n_episodes):
        obs = env.reset()
        R = 0
        l = 0

        for t in range(horizon):
            action = policy(obs)
            obs, rew, done, infos = env.step(action)

            R += rew
            l += 1

            if done:
                Rs.append(R)
                lens.append(l)
                R = 0
                l = 0
                break

    return

class Runner:
    def __init__(
            self,
            env_name,
            policy_type,
            search_sampler,
            n_trials,
            n_episodes,
            play_best,
            ret_chart,
            ent_chart,
            env_seed
    ):

        self.env_name = env_name
        self.policy_type = policy_type
        self.search_sampler = search_sampler
        self.n_trials = n_trials
        self.n_episodes = n_episodes
        self.play_best = play_best
        self.horizon = 1000
        self.env_seed = env_seed

        self.best_yet = 0
        self.returns = []
        self.states_visited = []
        self.ret_chart = ret_chart
        self.ent_chart = ent_chart
        self.actions = []
        self.last_100_actions = deque([], maxlen=100)
        self.last_100_states = deque([], maxlen=100)


    def run_policy(self, policy, env_name, n_episodes=5, T=1000):
        env = gym.make(env_name)
        env.seed(self.env_seed)

        Rs = []
        lens = []
        acts = []
        ents = []
        obses = []
        rews = []
        for n in range(n_episodes):
            obs = env.reset()
            R = 0
            l = 0

            for t in range(T):
                action = np.clip(policy(obs), env.action_space.low, env.action_space.high)
                obs, rew, done, infos = env.step(action)
                acts.append(action)
                obses.append(obs)
                rews.append(rew)
                self.states_visited.append(obs)
                self.actions.append(action)
                self.last_100_states.append(obs)
                self.last_100_actions.append(action)

                if self.policy_type == "dense_gaussian":
                    curr_ent = multivariate_normal(mean=action, cov=policy.vs).entropy()
                    ents.append(curr_ent)

                R += rew
                l += 1

                if done:
                    Rs.append(R)
                    lens.append(l)
                    R = 0
                    l = 0
                    break

        if np.mean(Rs) > self.best_yet:
            self.best_yet = np.mean(Rs)
            self.best_rets = Rs
            self.best_actions = acts
            self.best_obs = obses
            self.best_rew = rews

        self.returns.append(np.mean(Rs))
        self.ret_chart.add_rows([np.mean(Rs)])
        if self.policy_type == "dense_gaussian":
            self.ent_chart.add_rows([np.mean(ents)])
        return np.mean(Rs)

    def optimize_gdense_policy(self, trial, env_name, n_episodes=5, T=10000, min_param=-3., max_param=3., var_max=3.):
        env = gym.make(env_name)

        kernel = np.zeros((env.action_space.shape[0], env.observation_space.shape[0]))
        vs = np.zeros(env.action_space.shape[0])
        ctr = 0

        for i in range(env.action_space.shape[0]):
            vs[i] = trial.suggest_float("vars"+str(i), 0, var_max)
            for j in range(env.observation_space.shape[0]):
                kernel[i,j] = trial.suggest_float("kernel"+str(ctr), min_param, max_param)
                ctr += 1

        policy = DenseGaussianPolicy(kernel, vs, env)
        result = self.run_policy(policy, env_name, n_episodes=n_episodes, T=T)

        return result

    def optimize_dense_policy(self, trial, env_name, n_episodes=5, T=10000, min_param=-3., max_param=3., var_max=3.):
        env = gym.make(env_name)

        kernel = np.zeros((env.action_space.shape[0], env.observation_space.shape[0]))
        ctr = 0

        for i in range(env.action_space.shape[0]):
            for j in range(env.observation_space.shape[0]):
                kernel[i,j] = trial.suggest_float("kernel"+str(ctr), min_param, max_param)
                ctr += 1

        policy = DensePolicy(kernel, env)
        result = self.run_policy(policy, env_name, n_episodes=n_episodes, T=T)

        return result

    def train(self):
        allowed_samplers = ("cmaes", "tpe", "random")
        allowed_policies = ("dense_gaussian", "dense", "sdense", "sgdense")
        assert self.search_sampler in allowed_samplers, f"{search_sampler} not supported. Pick one of {allowed_samplers}"
        assert self.policy_type in allowed_policies, f"{policy_type} not supported. Pick one of {allowed_policies}"

        if self.search_sampler == "cmaes":
            sampler = optuna.samplers.CmaEsSampler()
        elif self.search_sampler == "tpe":
            sampler = optuna.samplers.TPESampler()
        elif self.search_sampler == "random":
            sampler = optuna.samplers.RandomSampler()
        else:
            sampler = optuna.samplers.CmaEsSampler()

        if self.policy_type == "dense_gaussian":
            pol_fcn = self.optimize_gdense_policy
            policy = DenseGaussianPolicy
        elif self.policy_type == "dense":
            pol_fcn = self.optimize_dense_policy
            policy = DensePolicy
        else:
            raise ValueError(f"Picked unsupported policy! Available options are {allowed_policies}")

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda trial: pol_fcn(trial, self.env_name, n_episodes=self.n_episodes, T=self.horizon), n_trials=self.n_trials)

        best_params = study.best_params

        return policy, best_params

def plot_actions(actions, rew, obs):
    mu_act = np.mean(actions)
    std_act = np.std(actions)
    acts = np.asarray([float(a) for a in actions]).squeeze()
    rets = np.asarray([float(r) for r in rew]).squeeze()
    data = pd.DataFrame({"Action taken": acts})
    chart = alt.Chart(data).mark_bar().encode(
        alt.X("Action taken", bin=True, axis=alt.Axis(grid=True)),
        y="count()"
    )
    states = np.asarray(obs).squeeze()
    rews = np.asarray(rew).squeeze()
    heatmap_data = pd.DataFrame({"Cart X position": states[:,0], "Cart X velocity": states[:, 1], "Action taken": acts, "Reward earned": rews, "Cosine of Pole angle": states[:,2], "Pole angular velocity": states[:,4]})

    heatmap1 = alt.Chart(heatmap_data).mark_rect().encode(
        alt.X("Cart X position:Q", bin=True, axis=alt.Axis(grid=True)),
        alt.Y("Cart X velocity:Q", bin=True, axis=alt.Axis(grid=True)),
        alt.Color("Action taken:Q", scale=alt.Scale(scheme="greenblue"))
    ).interactive()

    heatmap2 = alt.Chart(heatmap_data).mark_rect().encode(
        alt.X("Cosine of Pole angle", bin=True, axis=alt.Axis(grid=True)),
        alt.Y("Pole angular velocity", bin=True, axis=alt.Axis(grid=True)),
        alt.Color("Action taken", scale=alt.Scale(scheme="greenblue"))
    ).interactive()



@click.command()
@click.option("--env-name", "-env", type=str, default="InvertedPendulumBulletEnv-v0")
@click.option("--n-trials", "-trials", type=int, default=100)
@click.option("--n-episodes", "-neps", type=int, default=5)
@click.option("--horizon", "-t", type=int, default=1000)
@click.option("--search-sampler", "-search", type=str, default="cma")
@click.option("--policy-type", "-policy", type=str, default="dense")
@click.option("--save-params", "-save", type=bool, default=False)
@click.option("--play-best", "-play", type=bool, default=True)
@click.option("--n-eval-episodes", "-neval", type=int, default=100)
def train(env_name, n_trials, n_episodes, horizon, search_sampler, policy_type, save_params, play_best, n_eval_episodes):
    allowed_samplers = ("cma", "tpe", "random")
    allowed_policies = ("gdense", "dense", "sdense", "sgdense")
    assert search_sampler in allowed_samplers, f"{search_sampler} not supported. Pick one of {allowed_samplers}"
    assert policy_type in allowed_policies, f"{policy_type} not supported. Pick one of {allowed_policies}"

    if search_sampler == "cma":
        sampler = optuna.samplers.CmaEsSampler()
    elif search_sampler == "tpe":
        sampler = optuna.samplers.TPESampler()
    elif search_sampler == "random":
        sampler = optuna.samplers.RandomSampler()
    else:
        sampler = optuna.samplers.CmaEsSampler()

    if policy_type == "gdense":
        pol_fcn = optimize_gdense_policy
        policy = DenseGaussianPolicy
    elif policy_type == "dense":
        pol_fcn = optimize_dense_policy
        policy = DensePolicy
    elif policy_type == "sgdense":
        pol_fcn = optimize_sdense_policy
        policy = SigmoidDenseGaussianPolicy
    elif policy_type == "sdense":
        pol_fcn = optimize_sdense_policy
        policy = SigmoidDensePolicy
    else:
        raise ValueError(f"Picked unsupported policy! Available options are {allowed_policies}")

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: pol_fcn(trial, env_name, n_episodes=n_episodes, T=horizon), n_trials=n_trials)

    best_params = study.best_params

    if save_params:
        import pickle as pkl
        import os
        path = f"params/{env_name}/{policy_type}/{int(time.time())}"
        os.makedirs(path, exist_ok=True)
        with open(path+"params.pkl", "wb") as f:
            pkl.dump(best_params, f)

    if play_best:
        path = f"videos/{env_name}/{policy_type}/{int(time.time())}/"
        video_rollout(policy, best_params, env_name, n_eval_episodes, horizon, path)

if __name__ == "__main__":
    train()
