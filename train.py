import gymnasium
from stable_baselines3 import PPO, DDPG, SAC, A2C, TD3
from stable_baselines3.common.callbacks import EvalCallback
import os
import utils
from gymnasium.envs.registration import register
import config
import shutil
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, StopTrainingOnNoModelImprovement
from earlystopping import EarlyStoppingCallback
import torch


register(
    id="reactor_v2",
    entry_point="reactor_env:Reactor",
    kwargs={'experiment_name':"default"}
)

# Experiment Details
experiment_name = config.EXPERIMENT_NAME

# Model Specifications
model_name = config.MODEL

# Logfiles and model save path
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
utils.copy_files(files)
shutil.copy('config.py', f"experiments/{config.EXPERIMENT_NAME}")
shutil.copy('utils.py', f"experiments/{config.EXPERIMENT_NAME}")

# Compile the environment
env = gymnasium.make('reactor_v2', experiment_name=experiment_name,scr_opt=config.OPT_SUB_CELL_RATIO, mue_opt = config.MUE_OPT, eval_model=False)
env.reset()

eval_env = gymnasium.make('reactor_v2', experiment_name=experiment_name, scr_opt=config.OPT_SUB_CELL_RATIO, mue_opt = config.MUE_OPT, eval_model=True)
eval_env.reset()

# Initiate the model dynamically based on config.MODEL
if model_name == "PPO":
    model = PPO('MlpPolicy', env, tensorboard_log=logdir, device='cuda')

elif model_name == "SAC":
    model = SAC(
        'MlpPolicy', 
        env, 
        tensorboard_log=logdir, 
        device='cuda'
        # learning_rate = 2.0100866791609192e-05,
        # buffer_size = 127415,
        # learning_starts = 8942,
        # batch_size= 512,
        # tau = 0.0265783511470118,
        # gamma=0.920548515347066,
        # train_freq = 1,
        # gradient_steps =  20,
        # target_entropy = -0.7978268822546424,
        # target_update_interval =16
        )

stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=10, verbose=1)
                 
# Evaluation Callback
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path=models_dir,
    n_eval_episodes=50,
    eval_freq=10_000,
    verbose=1,
    deterministic= False,
    callback_after_eval= stop_train_callback
)

# Training loop
TIMESTEPS = 5e6
#TIMESTEPS = 100_000
EPOCHS = 1
model_save_path = os.path.join(models_dir,f'{experiment_name}.zip')
print(f"-------------------- Running Experiment: {experiment_name} -------------------- ")
print(f"-------------------- TRAINING {model_name} --------------------")
for i in range(1, EPOCHS + 1):
    print(f"Training {i}/{EPOCHS}")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=experiment_name, callback=eval_callback , progress_bar=True)
    model.save(model_save_path)