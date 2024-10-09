import gymnasium
import reactor_env
import register_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C , DDPG, HER, DQN, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import os
import config


model = "PPO"
experiment_name = "max_sub-0.010"
models_dir = f'models/{model}/{experiment_name}'
logdir = 'logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = gymnasium.make('reactor_v2')
env.reset()


model = PPO('MlpPolicy', env, tensorboard_log=logdir, device = 'cuda')

eval_callback = EvalCallback(
    env, 
    best_model_save_path=models_dir,
    n_eval_episodes = 20,
    eval_freq= 50_000,
    verbose=1
    )

TIMESTEPS = 100_000
EPOCHS = 10

for i in range(1,EPOCHS+1):
    print(f"Training {i}/{EPOCHS}")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=experiment_name, callback= eval_callback, progress_bar=True)
 