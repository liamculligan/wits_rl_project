Reinforcement Learning for Power Grid Management

This project trains reinforcement learning agents to manage power grids using the Grid2Op environment.

### Requirements

pip install -r requirements.txt

### Execution
Run the script to start training and evaluation:

grid_experiments.py


### Main Function Arguments
- `iteration`: Experiment number for record-keeping
- `use_ppo`, `use_a2c`, `use_sac`: Include PPO, A2C, or SAC algorithms
- `use_random`, `use_do_nothing`: Test random and do-nothing baselines
- `reduced_observation_space`, `reduced_action_space`: Enable reduced observation/action spaces
- `use_reward_shaping`: Whether to use reward shaping
- `num_test_episodes`: Test episodes for evaluation
- `total_timesteps`: Training steps for each model
- `check_freq`: Training check frequency
- `window_size`: Rolling window size for early stopping
- `patience`: Early stopping patience
- `min_improvement`: Min improvement threshold for early stopping

### Note
Adjust directory paths as needed for your environment.
