import os
import gymnasium
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.envs.registration import register
import config
import utils
import shutil

# Register the custom reactor environment
register(
    id="reactor_v2",
    entry_point="reactor_env:Reactor",
    kwargs={'experiment_name': config.EXPERIMENT_NAME}
)

# Experiment and model details from config.py
experiment_name = config.EXPERIMENT_NAME
model_name = config.MODEL

# Logfiles and model save path
models_dir = f'experiments/{experiment_name}/model'
logdir = f'logs'

# Create necessary directories if they don't exist
os.makedirs(experiment_name, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Copy essential scripts to experiment folder for reproducibility
files = os.listdir("copy_scripts")
utils.copy_files(files)
shutil.copy('config.py', config.EXPERIMENT_NAME)

# Initialize the environment
env = gymnasium.make('reactor_v2', experiment_name=experiment_name)
env.reset()

# Choose the model dynamically based on config.MODEL
model = PPO(
    'MlpPolicy', 
    env, 
    learning_rate=1e-4, 
    n_steps=4096, 
    batch_size=512, 
    n_epochs=15, 
    gamma=0.995, 
    clip_range=0.1, 
    gae_lambda=0.9, 
    ent_coef=0.01, 
    tensorboard_log=logdir, 
    device='cuda'
    )


# Setup evaluation callback
eval_callback = EvalCallback(
    env,
    best_model_save_path=models_dir,
    n_eval_episodes=20,
    eval_freq=50_000,
    verbose=1
)

# Training loop details
TIMESTEPS = 100_000  # Timesteps for each training epoch
EPOCHS = 10          # Number of epochs to train

# Save path for the model
model_save_path = os.path.join(models_dir, f'{experiment_name}.zip')

# Start the training process
print(f"-------------------- Running Experiment: {experiment_name} -------------------- ")
print(f"-------------------- TRAINING {model_name} --------------------")

for i in range(1, EPOCHS + 1):
    print(f"Training {i}/{EPOCHS}")
    model.learn(
        total_timesteps=TIMESTEPS, 
        reset_num_timesteps=False, 
        tb_log_name=experiment_name, 
        callback=eval_callback, 
        progress_bar=True
    )
    # Save the model after each epoch
    model.save(model_save_path)

print("Training complete. Final model saved at:", model_save_path)
