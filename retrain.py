import gymnasium
from stable_baselines3 import PPO, DDPG, SAC, A2C, TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import os
import utils
from gymnasium.envs.registration import register
import config
import shutil
import numpy as np
from earlystopping import EarlyStoppingCallback
from eval_callback_new import EpisodeBasedEvalCallback



register(
    id="reactor_v2",
    entry_point="reactor_env:Reactor",
    kwargs={'experiment_name':"default"}
)

# Experiment Details
scr = 0.002
mue_opt = 0.10
experiment_name = f"sac_rt_scr-{scr}_mopt-{mue_opt}"

# Model Specifications
model_name = 'SAC'

# Logfiles and model save path
trained_model_name = 'sac_evalcb'
trained_model_path = os.path.join('experiments', trained_model_name, 'model', 'best_model.zip')
models_dir = f'experiments/{experiment_name}/model'
logdir = f'logs'

# Create paths if they don't exist
if not os.path.exists(f"experiments/{experiment_name}"):
    os.makedirs(f"experiments/{experiment_name}")

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Copy files to experiment folder
files = os.listdir("copy_scripts")
for file in files:
    file_path = os.path.join('copy_scripts', file)
    shutil.copy(file_path, f"experiments/{experiment_name}")
shutil.copy('config.py', f"experiments/{experiment_name}")
shutil.copy('utils.py', f"experiments/{experiment_name}")

# Compile the environment
env = gymnasium.make('reactor_v2', experiment_name=experiment_name, scr_opt= scr, mue_opt = mue_opt, eval_model=False)
env.reset()

env_test = gymnasium.make('reactor_v2', experiment_name=experiment_name, scr_opt= scr, mue_opt = mue_opt, eval_model=True)
env_test.reset()

model = SAC.load(trained_model_path, env=env, device='cuda')  # Load the pretrained model
model.set_env(env)

from episode_count_callback import EpisodeCounterCallback
# Initialize an empty list to store episode counts
episode_counts = []

# Create the callback and pass the list to it
episode_counter = EpisodeCounterCallback(episode_list=episode_counts)

stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=5, verbose=1)

enz_epi_eval_callback = EpisodeBasedEvalCallback(env_test, model_dir=models_dir ,eval_interval=150, n_eval_episodes=50, patience=30, verbose=1)

# Evaluation Callback
eval_callback = EvalCallback(
    env_test, 
    best_model_save_path=models_dir,
    n_eval_episodes=50,
    eval_freq=10_000,
    verbose=0,
    deterministic= False,
    callback_after_eval= stop_train_callback
)
earlystopping_callback = EarlyStoppingCallback(patience=10000)


TIMESTEPS = 5e6
EPOCHS = 1
model_save_path = os.path.join(models_dir,f'{experiment_name}.zip')
print(f"-------------------- Running Experiment: {experiment_name} -------------------- ")
print(f"-------------------- TRAINING {model_name} --------------------")
for i in range(1, EPOCHS + 1):
    print(f"Training {i}/{EPOCHS}")
    model.learn(
        total_timesteps=TIMESTEPS, 
        reset_num_timesteps=False, 
        tb_log_name=experiment_name,
        callback=[eval_callback, episode_counter], 
        progress_bar=True
    )

print("------------ Retraining Complete ! ------------")
