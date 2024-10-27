import gymnasium as gym
import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from gymnasium.spaces import Discrete, Box
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from lightsim2grid import LightSimBackend
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
from typing import Dict, Literal, Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from google.colab import drive
from tqdm import tqdm
import time
import csv
import random

class Gym2OpEnv(gym.Env):
    def __init__(self, env_config: Dict[Literal["backend_cls", "backend_options", "env_name", "env_is_test"], Any] = None,
                 reduced_observation_space=False, reduced_action_space=False, use_reward_shaping=True):
        super().__init__()
        self.reduced_observation_space = reduced_observation_space
        self.reduced_action_space = reduced_action_space
        self.use_reward_shaping = use_reward_shaping

        if env_config is None:
            env_config = {}

        backend_cls = env_config.get("backend_cls", LightSimBackend)
        backend_options = env_config.get("backend_options", {})
        self._backend = backend_cls(**backend_options)

        self._env_name = env_config.get("env_name", "l2rpn_case14_sandbox")
        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
        p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep

        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=env_config.get("env_is_test", False),
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p)

        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        cr.initialize(self._g2op_env)

        self._gym_env = gym_compat.GymEnv(self._g2op_env)
        self.setup_observations()
        self.setup_actions()

        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space

        # Initialise tracking vars for reward shaping
        self.previous_rho_max = None
        self.consecutive_improvements = 0
        self.episode_step = 0

    def setup_observations(self):
        if self.reduced_observation_space:
            obs_attr_to_keep = ["rho", "gen_p", "load_p", "topo_vect", "v_or", "v_ex"]
            self._gym_env.observation_space = gym_compat.BoxGymObsSpace(
                self._g2op_env.observation_space,
                attr_to_keep=obs_attr_to_keep)
        else:
            self._gym_env.observation_space = gym_compat.BoxGymObsSpace(
                self._g2op_env.observation_space)
        self.observation_space = self._gym_env.observation_space

    def setup_actions(self):
        if self.reduced_action_space:
            act_attr_to_keep = ["set_bus", "change_bus"]
            self._gym_env.action_space = gym_compat.MultiDiscreteActSpace(
                self._g2op_env.action_space,
                attr_to_keep=act_attr_to_keep)
        else:
            self._gym_env.action_space = gym_compat.BoxGymActSpace(
                self._g2op_env.action_space)
        self.action_space = self._gym_env.action_space

    def get_shaped_reward(self, original_reward, info):
        """Calculate shaped reward component separately"""
        shaped_component = 0

        # Get current maximum line capacity usage
        current_rho_max = max(info.get("rho", [0]))

        # Reward shaping based on line capacity improvements
        if self.previous_rho_max is not None:
            # Reward for reducing maximum line usage
            if current_rho_max < self.previous_rho_max:
                improvement = self.previous_rho_max - current_rho_max
                # Small bonus for improvement
                shaped_component += 0.1 * improvement
                self.consecutive_improvements += 1

                # Additional bonus for sustained improvements
                if self.consecutive_improvements > 2:
                    # Small bonus for consistent improvement
                    shaped_component += 0.05
            else:
                self.consecutive_improvements = 0

            # Penalty for very high line usage
            # Near critical threshold
            if current_rho_max > 0.95:
                shaped_component -= 0.1 * (current_rho_max - 0.95)

        # Early game incentive
        # Early in episode
        if self.episode_step < 50:
            if current_rho_max < 0.8:
                shaped_component += 0.05

        self.previous_rho_max = current_rho_max

        return original_reward + shaped_component, shaped_component

    def step(self, action):

        step_result = self._gym_env.step(action)

        # Version issues
        if len(step_result) == 4:
            observation, original_reward, done, info = step_result
            info = info or {}
        else:
            observation, original_reward, terminated, truncated, info = step_result
            done = terminated or truncated
            info = info or {}

        self.episode_step += 1

        # Calculate shaped reward and store it
        if self.use_reward_shaping:
            total_reward, shaped_component = self.get_shaped_reward(original_reward, info)
            info['original_reward'] = original_reward
            info['shaped_component'] = shaped_component
            reward = total_reward
        else:
            reward = original_reward
            info['original_reward'] = original_reward
            info['shaped_component'] = 0

        if len(step_result) == 4:
            return observation, reward, done, info
        else:
            return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):

        self.previous_rho_max = None
        self.consecutive_improvements = 0
        self.episode_step = 0
        return self._gym_env.reset(seed=seed, options=options)

class LoggingCallback(BaseCallback):
    def __init__(self, log_file, check_freq=20, window_size=100, patience=200, min_improvement=0.02):
        super().__init__()
        self.log_file = log_file
        self.timesteps = []
        self.current_episode_reward = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.check_freq = check_freq
        self.window_size = window_size
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_median = None
        self.steps_without_improvement = 0
        self.current_episode_steps = 0

        ensure_directory_exists(os.path.dirname(self.log_file))
        self.csv_file = open(self.log_file, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['episode', 'timestep', 'episode_reward', 'episode_length',
                            'rolling_median_20', 'rolling_median_100',
                            'rolling_mean_20', 'rolling_mean_100'])

    def _on_step(self):
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_steps += 1

        if self.locals.get('dones')[0]:
            self.timesteps.append(self.num_timesteps)
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_steps)

            # Calculate rolling stats
            if len(self.episode_rewards) >= 20:
                rolling_median_20 = float(np.median(self.episode_rewards[-20:]))
                rolling_mean_20 = float(np.mean(self.episode_rewards[-20:]))
            else:
                rolling_median_20 = float(self.current_episode_reward)
                rolling_mean_20 = float(self.current_episode_reward)

            if len(self.episode_rewards) >= 100:
                rolling_median_100 = float(np.median(self.episode_rewards[-100:]))
                rolling_mean_100 = float(np.mean(self.episode_rewards[-100:]))
            else:
                rolling_median_100 = float(self.current_episode_reward)
                rolling_mean_100 = float(self.current_episode_reward)

            self.writer.writerow([
                len(self.episode_rewards),
                self.num_timesteps,
                self.current_episode_reward,
                self.current_episode_steps,
                rolling_median_20,
                rolling_median_100,
                rolling_mean_20,
                rolling_mean_100])

            self.csv_file.flush()

            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_steps = 0

            # Early stopping check using median
            if len(self.episode_rewards) >= self.window_size:
                current_median = np.median(self.episode_rewards[-self.window_size:])

                if self.best_median is None:
                    self.best_median = current_median
                    return True

                relative_improvement = (current_median - self.best_median)/abs(self.best_median)

                print(f"\nEpisode: {len(self.episode_rewards)}")
                print(f"Current median episode reward: {current_median:.4f}")
                print(f"Best median episode reward: {self.best_median:.4f}")
                print(f"Relative improvement: {relative_improvement:.4f}")
                print(f"Episodes without improvement: {self.steps_without_improvement}")

                if relative_improvement > self.min_improvement:
                    self.best_median = current_median
                    self.steps_without_improvement = 0

                else:
                    self.steps_without_improvement += 1

                if self.steps_without_improvement >= self.patience:
                    print(f"\nEarly stopping after {len(self.episode_rewards)} episodes")
                    print(f"No improvement for {self.patience} episodes")
                    print(f"Best median episode reward: {self.best_median:.4f}")
                    self.csv_file.close()
                    return False

        return True

    def _on_training_end(self):
        if not self.csv_file.closed:
            self.csv_file.close()

def make_env(reduced_obs=False, reduced_acts=False, use_reward_shaping=True):
    return lambda: Gym2OpEnv(reduced_observation_space=reduced_obs,
                            reduced_action_space=reduced_acts,
                            use_reward_shaping=use_reward_shaping)

def create_run_directory(base_dir,
                         iteration):
    run_dir = os.path.join(base_dir, f"iteration_{iteration}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def generate_test_seeds(num_test_episodes):
    return list(range(50000, 50000 + num_test_episodes))

def train_without_hyperopt(env,
                           model_class,
                           total_timesteps,
                           log_file,
                           check_freq=20,
                           window_size=100,
                           patience=200,
                           min_improvement=0.02):

    seed = 44
    model = model_class("MlpPolicy", env, verbose=0, seed=seed)

    logging_callback = LoggingCallback(
        log_file=log_file,
        check_freq=check_freq,
        window_size=window_size,
        patience=patience,
        min_improvement=min_improvement)

    model.learn(total_timesteps=total_timesteps, callback=logging_callback)

    # Generate plots and save data
    plot_file = log_file.replace('.csv', '_progress_plot.png')
    plot_training_progress(log_file, plot_file)

    processed_file = log_file.replace('.csv', '_processed.csv')
    save_training_progress(log_file, processed_file)

    return model

def evaluate_all_agents(env,
                        agents,
                        num_episodes,
                        seeds=None):
    """
    Evaluate multiple agents on consistent envs
    """
    if seeds is None:
        raise ValueError("Seeds must be provided for eval")

    results = {name: {
        "returns": [],
        "survivals": [],
        "survival_timesteps": [],
        "percent_overcapacity": []} for name in agents.keys()}

    for episode in range(num_episodes):

        seed = int(seeds[episode])
        print(f"\nEpisode {episode + 1}/{num_episodes} (seed: {seed})")

        # Need initial state and chronic
        env.reset(seed=seed)
        chronic_id = env._g2op_env.chronics_handler.get_id()
        print(f"Using chronic ID: {chronic_id}")

        for name, agent in agents.items():

            print(f"\nTesting {name}...")

            # Reset to same state
            env._g2op_env.seed(seed)
            env._g2op_env.chronics_handler.tell_id(chronic_id)
            obs, _ = env.reset()

            done = False
            episode_return = 0
            step_count = 0
            overcapacity_instances = 0
            total_checks = 0

            while not done:

                if isinstance(agent, (PPO, A2C, SAC)):
                    action, _ = agent.predict(obs, deterministic=True)

                # Do nothing agent
                elif agent is None:  #
                    action = env.action_space.sample()
                    action[:] = 0
                 # Random agent
                else:
                    action = env.action_space.sample()

                step_result = env.step(action)

                # Env consistency issues
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                elif len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    raise ValueError("Unexpected step result length")

                episode_return += reward
                step_count += 1

                if "rho" in info:
                    current_overcap = np.sum(info["rho"] > 1.0)
                    overcapacity_instances += current_overcap
                    total_checks += len(info["rho"])

            print(f"Return: {episode_return:.2f}")
            print(f"Steps: {step_count}")

            results[name]["returns"].append(episode_return)
            results[name]["survivals"].append(
                1 if step_count == env._g2op_env.chronics_handler.max_timestep() else 0)
            results[name]["survival_timesteps"].append(step_count)
            results[name]["percent_overcapacity"].append(
                (overcapacity_instances/total_checks * 100) if total_checks > 0 else 0)

    return results

def compute_statistics(results):
    stats = {}

    for name, data in results.items():
        stats[name] = {
            "mean_reward": float(np.mean(data["returns"])),
            "median_reward": float(np.median(data["returns"])),
            "std_reward": float(np.std(data["returns"])),
            "survival_rate": float(np.mean(data["survivals"])),
            "survival_rate_std": float(np.std(data["survivals"])),
            "avg_survival_timesteps": float(np.mean(data["survival_timesteps"])),
            "median_survival_timesteps": float(np.median(data["survival_timesteps"])),
            "avg_survival_timesteps_std": float(np.std(data["survival_timesteps"])),
            "percent_overcapacity": float(np.mean(data["percent_overcapacity"])),
            "percent_overcapacity_std": float(np.std(data["percent_overcapacity"]))}

    return stats

def plot_training_progress(log_file,
                           plot_file):

    df = pd.read_csv(log_file)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(df['episode'], df['episode_reward'], alpha=0.3, label='Episode Reward', color='gray')

    rolling_med = df['episode_reward'].rolling(window=100, min_periods=100).median()
    valid_med_mask = ~rolling_med.isna()

    if valid_med_mask.any():
        ax1.plot(df.loc[valid_med_mask, 'episode'], rolling_med[valid_med_mask],
                label='Rolling Median', color='blue', linewidth=2)

    ax1.axvline(x=df['episode'].iloc[-1], color='r', linestyle='--',
                label='Early Stopping Point')

    ax1.set_title('Training Progress (Episode Rewards)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(df['episode'], df['episode_length'], alpha=0.3, label='Episode Length', color='gray')

    rolling_med_len = df['episode_length'].rolling(window=100, min_periods=100).median()

    valid_med_mask = ~rolling_med_len.isna()

    if valid_med_mask.any():
        ax2.plot(df.loc[valid_med_mask, 'episode'], rolling_med_len[valid_med_mask],
                label='Rolling Median', color='blue', linewidth=2)

    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps per Episode')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

def save_training_progress(log_file,
                           output_file):

    df = pd.read_csv(log_file)
    df.to_csv(output_file, index=False)

def create_test_evaluation_plots(stats,
                                 combined_results_dir,
                                 raw_data):

    agents = list(stats.keys())

    # Mean Reward Bar Plot
    plt.figure(figsize=(12, 6))
    means = [stats[agent]["mean_reward"] for agent in agents]
    stds = [stats[agent]["std_reward"] for agent in agents]
    plt.bar(agents, means, yerr=stds, capsize=10, color='lightblue', alpha=0.7)
    plt.title('Mean Test Episode Reward by Agent')
    plt.xlabel('Agent')
    plt.ylabel('Average Total Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(combined_results_dir, 'agent_comparison_mean_reward.png'))
    plt.close()

    # Median Reward Bar Plot
    plt.figure(figsize=(12, 6))
    medians = [stats[agent]["median_reward"] for agent in agents]
    plt.bar(agents, medians, capsize=10, color='lightblue', alpha=0.7)
    plt.title('Median Test Episode Reward by Agent')
    plt.xlabel('Agent')
    plt.ylabel('Median Total Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(combined_results_dir, 'agent_comparison_median_reward.png'))
    plt.close()

    # Reward Boxplot
    plt.figure(figsize=(12, 6))
    reward_data = [raw_data[agent]["returns"] for agent in agents]
    bp = plt.boxplot(reward_data, labels=agents, patch_artist=True)

    for element in ['boxes', 'whiskers', 'fliers', 'caps']:
        plt.setp(bp[element], color='black')

    for box in bp['boxes']:
        box.set_facecolor('lightblue')
        box.set_alpha(0.7)

    plt.setp(bp['medians'], color='red', linewidth=2)
    plt.setp(bp['fliers'], marker='o', markerfacecolor='gray', alpha=0.5, markersize=4)

    plt.title('Test Episode Reward Distribution by Agent')
    plt.xlabel('Agent')
    plt.ylabel('Episode Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(combined_results_dir, 'agent_comparison_reward_boxplot.png'))
    plt.close()

    # Survival Rate Bar Plot
    plt.figure(figsize=(12, 6))
    survival_rates = [stats[agent]["survival_rate"] for agent in agents]
    survival_rate_stds = [stats[agent]["survival_rate_std"] for agent in agents]
    plt.bar(agents, survival_rates, yerr=survival_rate_stds, capsize=10, color='lightblue', alpha=0.7)
    plt.title('Mean Test Survival Rate by Agent')
    plt.xlabel('Agent')
    plt.ylabel('Survival Rate')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(combined_results_dir, 'agent_comparison_survival.png'))
    plt.close()

    # Mean Survival Bar Plot
    plt.figure(figsize=(12, 6))
    avg_survival_timesteps = [stats[agent]["avg_survival_timesteps"] for agent in agents]
    avg_survival_timesteps_stds = [stats[agent]["avg_survival_timesteps_std"] for agent in agents]
    plt.bar(agents, avg_survival_timesteps, yerr=avg_survival_timesteps_stds, capsize=10, color='lightblue', alpha=0.7)
    plt.title('Mean Test Episode Survival Time by Agent')
    plt.xlabel('Agent')
    plt.ylabel('Average Timesteps Until Episode End')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(combined_results_dir, 'agent_comparison_mean_survival_time.png'))
    plt.close()

    # Median Survival Time Br Plot
    plt.figure(figsize=(12, 6))
    median_survival_timesteps = [stats[agent]["median_survival_timesteps"] for agent in agents]
    plt.bar(agents, median_survival_timesteps, capsize=10, color='lightblue', alpha=0.7)
    plt.title('Median Test Survival Time by Agent')
    plt.xlabel('Agent')
    plt.ylabel('Median Timesteps Until Episode End')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(combined_results_dir, 'agent_comparison_median_survival_time.png'))
    plt.close()

    # Survival Time boxplt
    plt.figure(figsize=(12, 6))
    survival_data = [raw_data[agent]["survival_timesteps"] for agent in agents]
    bp = plt.boxplot(survival_data, labels=agents, patch_artist=True)

    for element in ['boxes', 'whiskers', 'fliers', 'caps']:
        plt.setp(bp[element], color='black')

    for box in bp['boxes']:
        box.set_facecolor('lightblue')
        box.set_alpha(0.7)

    plt.setp(bp['medians'], color='red', linewidth=2)
    plt.setp(bp['fliers'], marker='o', markerfacecolor='gray', alpha=0.5, markersize=4)

    plt.title('Test Episode Survival Time Distribution by Agent')
    plt.xlabel('Agent')
    plt.ylabel('Episode Survival Time (Timesteps)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(combined_results_dir, 'agent_comparison_survival_time_boxplot.png'))
    plt.close()

    # Percent Over Capacity
    plt.figure(figsize=(12, 6))
    percent_overcapacity = [stats[agent]["percent_overcapacity"] for agent in agents]
    percent_overcapacity_stds = [stats[agent]["percent_overcapacity_std"] for agent in agents]
    plt.bar(agents, percent_overcapacity, yerr=percent_overcapacity_stds, capsize=10, color='lightblue', alpha=0.7)
    plt.title('Mean Percent Over Capacity by Agent')
    plt.xlabel('Agent')
    plt.ylabel('Percent Over Capacity')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(combined_results_dir, 'agent_comparison_overcapacity.png'))
    plt.close()

def train_algorithm(algorithm,
                    run_dir,
                    num_train_envs,
                    total_timesteps,
                    check_freq,
                    window_size,
                    patience,
                    min_improvement,
                    reduced_observation_space,
                    reduced_action_space,
                    use_reward_shaping):

    seed = 44
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    algorithm_dir = os.path.join(run_dir, algorithm.lower())
    os.makedirs(algorithm_dir, exist_ok=True)

    log_file = os.path.join(algorithm_dir, f"{algorithm}_training_log.csv")
    print(f"Log file will be saved to: {log_file}")
    ensure_directory_exists(os.path.dirname(log_file))

    print(f"Creating {num_train_envs} training environment(s)...")
    train_env = DummyVecEnv([lambda: make_env(reduced_observation_space, reduced_action_space, use_reward_shaping)()
                            for _ in range(num_train_envs)])

    print("Training env created")

    model_class = PPO if algorithm == "PPO" else A2C if algorithm == "A2C" else SAC
    print(f"Starting training with {model_class.__name__}")

    print(f"Training parameters:")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Check frequency: {check_freq}")
    print(f"Window size: {window_size}")
    print(f"Patience: {patience}")
    print(f"Min improvement: {min_improvement}")
    print(f"Reduced observation space: {reduced_observation_space}")
    print(f"Reduced action space: {reduced_action_space}")
    print(f"Using reward shaping: {use_reward_shaping}")

    model = train_without_hyperopt(
        train_env,
        model_class,
        total_timesteps,
        log_file,
        check_freq=check_freq,
        window_size=window_size,
        patience=patience,
        min_improvement=min_improvement)

    model_path = os.path.join(algorithm_dir, f"{algorithm}_model")
    model.save(model_path)

    return {"model": model}

def save_test_results_csv(stats,
                          raw_results,
                          output_file,
                          raw_data_file):

    with open(output_file, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(['Agent', 'Mean Reward', 'Median Reward', 'Std Reward',
                        'Survival Rate', 'Survival Rate Std',
                        'Avg Survival Timesteps', 'Median Survival Timesteps',
                        'Avg Survival Timesteps Std',
                        'Percent Over Capacity', 'Percent Over Capacity Std'])

        for agent, data in stats.items():

            writer.writerow([
                agent,
                f"{data['mean_reward']:.2f}",
                f"{data['median_reward']:.2f}",
                f"{data['std_reward']:.2f}",
                f"{data['survival_rate']:.2f}",
                f"{data['survival_rate_std']:.2f}",
                f"{data['avg_survival_timesteps']:.2f}",
                f"{data['median_survival_timesteps']:.2f}",
                f"{data['avg_survival_timesteps_std']:.2f}",
                f"{data['percent_overcapacity']:.2f}",
                f"{data['percent_overcapacity_std']:.2f}"])

    raw_df = pd.DataFrame()
    for agent, data in raw_results.items():
        agent_df = pd.DataFrame({
            'Agent': [agent] * len(data['returns']),
            'Episode_Reward': data['returns'],
            'Survival_Time': data['survival_timesteps'],
            'Survived': data['survivals'],
            'Percent_Overcapacity': data['percent_overcapacity']})
        raw_df = pd.concat([raw_df, agent_df])

    raw_df.to_csv(raw_data_file, index=False)

def run_experiment(base_dir,
                   iteration,
                   use_ppo,
                   use_a2c,
                   use_sac,
                   use_random,
                   use_do_nothing,
                   reduced_observation_space,
                   reduced_action_space,
                   use_reward_shaping,
                   num_train_envs,
                   num_val_episodes,
                   num_test_episodes,
                   total_timesteps,
                   test_seeds,
                   check_freq,
                   window_size,
                   patience,
                   min_improvement):

    print("\nStarting New Experiment:")
    print(f"Base directory: {base_dir}")
    print(f"Iteration: {iteration}")
    print("Using algorithms:")
    print(f"PPO: {use_ppo}")
    print(f"A2C: {use_a2c}")
    print(f"SAC: {use_sac}")
    print(f"Random: {use_random}")
    print(f"Do Nothing: {use_do_nothing}")
    print("Space configuration:")
    print(f"Reduced observation space: {reduced_observation_space}")
    print(f"Reduced action space: {reduced_action_space}")
    print(f"Using reward shaping: {use_reward_shaping}")

    results = {}
    run_dir = create_run_directory(base_dir, iteration)

    # Train RL algorithms with reward shaping
    if use_ppo:
        print("\nTraining PPO:")
        results["PPO"] = train_algorithm("PPO", run_dir, num_train_envs, total_timesteps,
                                       check_freq, window_size, patience, min_improvement,
                                       reduced_observation_space, reduced_action_space,
                                       use_reward_shaping)

    if use_a2c:
        print("\nTraining A2C:")
        results["A2C"] = train_algorithm("A2C", run_dir, num_train_envs, total_timesteps,
                                       check_freq, window_size, patience, min_improvement,
                                       reduced_observation_space, reduced_action_space,
                                       use_reward_shaping)

    if use_sac:
        print("\nTraining SAC:")
        results["SAC"] = train_algorithm("SAC", run_dir, num_train_envs, total_timesteps,
                                       check_freq, window_size, patience, min_improvement,
                                       reduced_observation_space, reduced_action_space,
                                       use_reward_shaping)

    print("\nStarting eval:")
    print("Creating test env...")
    test_env = Gym2OpEnv(reduced_observation_space=reduced_observation_space,
                         reduced_action_space=reduced_action_space,
                         use_reward_shaping=False)  # Always False for testing!!!!!

    agents = {}
    if use_do_nothing:
        agents["Do Nothing"] = None
    if use_random:
        agents["Random"] = lambda obs: test_env.action_space.sample()

    # Add trained models if successfully trained
    if use_ppo and "PPO" in results:
        agents["PPO"] = results["PPO"]["model"]
    if use_a2c and "A2C" in results:
        agents["A2C"] = results["A2C"]["model"]
    if use_sac and "SAC" in results:
        agents["SAC"] = results["SAC"]["model"]

    print(f"Evaluating agents: {list(agents.keys())}")
    test_results = evaluate_all_agents(test_env, agents, num_test_episodes, test_seeds)
    test_stats = compute_statistics(test_results)

    print("\nSaving Results:")
    combined_results_dir = os.path.join(run_dir, "combined_results")
    os.makedirs(combined_results_dir, exist_ok=True)

    csv_file = os.path.join(combined_results_dir, 'test_results.csv')
    raw_data_file = os.path.join(combined_results_dir, 'test_results_raw.csv')
    save_test_results_csv(test_stats, test_results, csv_file, raw_data_file)

    create_test_evaluation_plots(test_stats, combined_results_dir, test_results)

    # Save training data for each agent
    for agent_name in results.keys():
        agent_dir = os.path.join(run_dir, agent_name.lower())
        training_data = pd.read_csv(os.path.join(agent_dir, f"{agent_name}_training_log.csv"))

        # Save episode-level data
        episode_data = pd.DataFrame({
            'episode': training_data['episode'],
            'episode_reward': training_data['episode_reward'],
            'episode_length': training_data['episode_length']})

        episode_data.to_csv(os.path.join(agent_dir, f"{agent_name}_episode_data.csv"), index=False)

    print("\nDone")
    return results, test_stats

def main(iteration,
         use_ppo=True,
         use_a2c=False,
         use_sac=False,
         use_random=True,
         use_do_nothing=True,
         reduced_observation_space=False,
         reduced_action_space=False,
         use_reward_shaping=True,
         num_train_envs=1,
         num_val_episodes=1,
         num_test_episodes=2,
         total_timesteps=1000,
         check_freq=20,
         window_size=100,
         patience=200,
         min_improvement=0.02,
         test_seed=888):

    drive.mount('/content/drive')

    base_dir = "/content/drive/My Drive/rl_project/model_runs"
    print(f"Creating base directory: {base_dir}")
    os.makedirs(base_dir, exist_ok=True)

    print(f"\nStarting experiment")
    print("Algorithms selected:")
    print(f"PPO: {use_ppo}")
    print(f"A2C: {use_a2c}")
    print(f"SAC: {use_sac}")
    print(f"Random: {use_random}")
    print(f"Do Nothing: {use_do_nothing}")
    print("Space configuration:")
    print(f"Reduced observation space: {reduced_observation_space}")
    print(f"Reduced action space: {reduced_action_space}")
    print(f"Using reward shaping: {use_reward_shaping}")

    print(f"Early stopping config (episode-based):")
    print(f"Check frequency: {check_freq} episodes")
    print(f"Window size: {window_size} episodes")
    print(f"Patience: {patience} episodes")
    print(f"Min improvement: {min_improvement*100}%")

    start_time = time.time()

    np.random.seed(test_seed)
    test_seeds = [int(x) for x in np.random.randint(0, 10000, size=num_test_episodes)]
    print(f"\nUsing test_seed: {test_seed}")
    print(f"Generated {len(test_seeds)} test seeds: {test_seeds}")

    results, test_stats = run_experiment(
        base_dir=base_dir,
        iteration=iteration,
        use_ppo=use_ppo,
        use_a2c=use_a2c,
        use_sac=use_sac,
        use_random=use_random,
        use_do_nothing=use_do_nothing,
        reduced_observation_space=reduced_observation_space,
        reduced_action_space=reduced_action_space,
        use_reward_shaping=use_reward_shaping,
        num_train_envs=num_train_envs,
        num_val_episodes=num_val_episodes,
        num_test_episodes=num_test_episodes,
        total_timesteps=total_timesteps,
        test_seeds=test_seeds,
        check_freq=check_freq,
        window_size=window_size,
        patience=patience,
        min_improvement=min_improvement)

    end_time = time.time()
    runtime_minutes = (end_time - start_time)/60

    print("\nTest Results:")
    for agent, metrics in test_stats.items():
        print(f"\n{agent}:")
        print(f"Median Reward: {metrics['median_reward']:.2f}")
        print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Survival Rate: {metrics['survival_rate']:.2f} ± {metrics['survival_rate_std']:.2f}")
        print(f"Avg Survival Timesteps: {metrics['avg_survival_timesteps']:.2f} ± {metrics['avg_survival_timesteps_std']:.2f}")
        print(f"Percent Over Capacity: {metrics['percent_overcapacity']:.2f}%")

    print(f"Total runtime: {runtime_minutes:.2f} minutes")

    run_dir = os.path.join(base_dir, f"iteration_{iteration}")
    seeds_file = os.path.join(run_dir, "test_seeds.txt")

    with open(seeds_file, "w") as f:
        f.write(f"Test seed: {test_seed}\n")
        f.write(f"Episode seeds: {test_seeds}")

    return results, test_stats

if __name__ == "__main__":

    results, stats = main(
        iteration=99,
        use_ppo=True,
        use_a2c=True,
        use_sac=False,
        use_random=False,
        use_do_nothing=False,
        reduced_observation_space=False, # Whether to use a reduced observation space
        reduced_action_space=False, # Whether to use a reduced action space
        use_reward_shaping=False, # Whether to use reward shaping
        num_train_envs=1, # Just keep as 1 - not implmented
        num_val_episodes=1, # Just keep as 1 - not implmented
        num_test_episodes=100, # Number of test episodes for evaluating agents
        total_timesteps=100000, #Total training timesteps for each model
        check_freq=20, # Frequency of training checks for logging
        window_size=20, # Size of the rolling window for median reward early stopping
        patience=20, # Early stopping patience
        min_improvement=0.05)