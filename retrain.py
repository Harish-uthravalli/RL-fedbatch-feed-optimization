import gymnasium
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import os
import shutil
import multiprocessing

from gymnasium.envs.registration import register
from earlystopping import EarlyStoppingCallback
from eval_callback_new import EpisodeBasedEvalCallback
from episode_count_callback import EpisodeCounterCallback

register(
    id="reactor_v2",
    entry_point="reactor_env:Reactor",
    kwargs={'experiment_name': "default"}
)

def run_experiment(scr, mue_opt):
    experiment_name = f"sac_rt_scr-{scr}_mopt-{mue_opt}"
    trained_model_name = 'sac_evalcb'
    trained_model_path = os.path.join('experiments', trained_model_name, 'model', 'best_model.zip')
    models_dir = f'experiments/{experiment_name}/model'
    logdir = 'logs'

    # Create necessary directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # Copy files to experiment folder
    files = os.listdir("copy_scripts")
    for file in files:
        shutil.copy(os.path.join('copy_scripts', file), f"experiments/{experiment_name}")
    shutil.copy('config.py', f"experiments/{experiment_name}")
    shutil.copy('utils.py', f"experiments/{experiment_name}")

    # Create environments
    env = gymnasium.make('reactor_v2', experiment_name=experiment_name, scr_opt=scr, mue_opt=mue_opt, eval_model=False)
    env.reset()
    env_test = gymnasium.make('reactor_v2', experiment_name=experiment_name, scr_opt=scr, mue_opt=mue_opt, eval_model=True)
    env_test.reset()

    # Load the pretrained model
    model = SAC.load(trained_model_path, env=env, device='cuda')
    model.set_env(env)

    # Initialize callbacks
    episode_counts = []
    episode_counter = EpisodeCounterCallback(episode_list=episode_counts)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=5, verbose=1)
    enz_epi_eval_callback = EpisodeBasedEvalCallback(env_test, model_dir=models_dir, eval_interval=150, n_eval_episodes=50, patience=30, verbose=1)
    eval_callback = EvalCallback(env_test, best_model_save_path=models_dir, n_eval_episodes=50, eval_freq=10_000, verbose=0, deterministic=False, callback_after_eval=stop_train_callback)
    earlystopping_callback = EarlyStoppingCallback(patience=10000)

    # Training loop
    TIMESTEPS = 5e6
    EPOCHS = 1
    print(f"-------------------- Running Experiment: {experiment_name} -------------------- ")
    print(f"-------------------- TRAINING SAC --------------------")

    for i in range(1, EPOCHS + 1):
        print(f"Training {i}/{EPOCHS} for {experiment_name}")
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name=experiment_name,
            callback=[eval_callback, episode_counter],
            progress_bar=True
        )

    print(f"------------ Retraining Complete for {experiment_name}! ------------")

if __name__ == "__main__":
    # Define different experiment configurations
    experiment_configs = [
        (0.002, 0.06),
        (0.0024444, 0.06),
        (0.00288889, 0.06),
        (0.0033333, 0.06),
        (0.00377778, 0.06),
        (0.00422222, 0.06),
        (0.00466667, 0.06),
        (0.00511111, 0.06),
        (0.00555556, 0.06),
        (0.006, 0.06)
    ]  # List of (scr, mue_opt) values
    run_experiment(0.006, 0.06)
