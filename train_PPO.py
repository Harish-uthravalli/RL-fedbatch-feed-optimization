import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import os
import utils
from gymnasium.envs.registration import register
import config
import shutil

# Set global seed for reproducibility
utils.set_global_seed(42)

# Register custom environment
register(
    id="reactor_v2",
    entry_point="reactor_env:Reactor",
    kwargs={'experiment_name': "default"}
)

# Experiment Details
experiment_name = config.EXPERIMENT_NAME
model_name = config.MODEL

# Directories for models and logs
models_dir = f'experiments/{experiment_name}/model'
logdir = 'logs'

# Create directories if they don’t exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Copy configuration and necessary files to experiment folder for reference
files = os.listdir("copy_scripts")
utils.copy_files(files)
shutil.copy('config.py', f"experiments/{experiment_name}")

# Initialize the environment
env = gymnasium.make('reactor_v2', experiment_name=experiment_name)
env.unwrapped.seed(42)  # Ensure environment's determinism by setting the seed

# Reset environment before training (optional but good practice)
env.reset(seed=42)

model = PPO(
    'MlpPolicy',
    env,
    tensorboard_log=logdir,
    device='cuda',
    learning_rate=0.00035,
    n_steps=3328,  # Adjusted n_steps to be divisible by batch_size (128)
    batch_size=128,
    n_epochs=4,
    gamma=0.96085,
    gae_lambda=0.9186,
    clip_range=0.2265,
    clip_range_vf=0.3671,
    ent_coef=0.00779,
    vf_coef=0.9642,
    max_grad_norm=0.8408,
    target_kl=0.063,
    seed=42
)

# Define evaluation callback for periodic evaluations during training
eval_callback = EvalCallback(
    env,
    best_model_save_path=models_dir,
    n_eval_episodes=20,
    eval_freq=50_000,
    verbose=1,
    deterministic=False  # Ensure deterministic behavior during evaluation
)

# Training loop parameters
TIMESTEPS = 100_000
EPOCHS = 10
model_save_path = os.path.join(models_dir, f'{experiment_name}.zip')

# Training loop
print(f"-------------------- Running Experiment: {experiment_name} -------------------- ")
print(f"-------------------- TRAINING {model_name} --------------------")
for i in range(1, EPOCHS + 1):
    print(f"Training epoch {i}/{EPOCHS}")
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=experiment_name,
        callback=eval_callback,
        progress_bar=True
    )
    # Save model checkpoint
    model.save(model_save_path)
