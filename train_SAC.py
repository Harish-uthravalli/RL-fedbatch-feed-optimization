import gymnasium
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.callbacks import EvalCallback
import os
import utils
from gymnasium.envs.registration import register
import config
import shutil


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
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Copy files to experiment folder
files = os.listdir("copy_scripts")
utils.copy_files(files)
shutil.copy('config.py', config.EXPERIMENT_NAME)

# Compile the environment
env = gymnasium.make('reactor_v2', experiment_name=experiment_name)
env.reset()

# Initiate the model dynamically based on config.MODEL
model = SAC(
    'MlpPolicy', 
    env, 
    learning_rate=5e-4, 
    buffer_size=int(2e6), 
    batch_size=512, 
    gradient_steps=2, 
    gamma=0.995, 
    tau=0.005, 
    ent_coef='auto', 
    target_entropy=-0.5 * env.action_space.shape[0], 
    learning_starts=10_000, 
    policy_kwargs=dict(net_arch=[256, 256]), 
    tensorboard_log=logdir, 
    device='cuda',
)

# Evaluation Callback
eval_callback = EvalCallback(
    env, 
    best_model_save_path=models_dir,
    n_eval_episodes=20,
    eval_freq=50_000,
    verbose=1
)

# Training loop
TIMESTEPS = 100_000
EPOCHS = 10
model_save_path = os.path.join(models_dir,f'{experiment_name}.zip')
print(f"-------------------- Running Experiment: {experiment_name} -------------------- ")
print(f"-------------------- TRAINING {model_name} --------------------")
for i in range(1, EPOCHS + 1):
    print(f"Training {i}/{EPOCHS}")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=experiment_name, callback=eval_callback, progress_bar=True)
    model.save(model_save_path)
