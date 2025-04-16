import gymnasium
from stable_baselines3 import PPO, DDPG, SAC, A2C, TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes, StopTrainingOnNoModelImprovement
import os
import shutil
import utils
from gymnasium.envs.registration import register
import config
from earlystopping import EarlyStoppingCallback
from eval_callback_new import EpisodeBasedEvalCallback
import torch
import numpy as np
import multiprocessing


def setup_experiment(experiment_name, model_name, scr, muopt):

    register(
        id="reactor_v2",
        entry_point="reactor_env:Reactor",
        kwargs={'experiment_name': experiment_name}
    )

    models_dir = f'experiments/{experiment_name}/model'
    logdir = 'logs'

    os.makedirs(f"experiments/{experiment_name}", exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    files = os.listdir("copy_scripts")
    utils.copy_files(files)
    shutil.copy('config.py', f"experiments/{experiment_name}")
    shutil.copy('utils.py', f"experiments/{experiment_name}")

    env = gymnasium.make('reactor_v2', experiment_name=experiment_name, scr_opt=scr, mue_opt=muopt, eval_model=False)
    env.reset()

    eval_env = gymnasium.make('reactor_v2', experiment_name=experiment_name, scr_opt=scr, mue_opt=muopt, eval_model=True)
    eval_env.reset()

    model = SAC('MlpPolicy', env, tensorboard_log=logdir, device='cuda')


    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=10, verbose=1)
    evaluate_enz_callback = EpisodeBasedEvalCallback(eval_env, experiment_name, scr_opt=scr, model_dir=models_dir, eval_interval=150, n_eval_episodes=50, patience=30, verbose=1)
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=10_000, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        n_eval_episodes=30,
        eval_freq=10_000,
        verbose=1,
        deterministic=False,
        callback_after_eval=stop_train_callback
    )

    TIMESTEPS = 5e6
    EPOCHS = 1
    model_save_path = os.path.join(models_dir, f'{experiment_name}.zip')

    print(f"-------------------- Running Experiment: {experiment_name} -------------------- ")
    print(f"-------------------- TRAINING {model_name} --------------------")
    for i in range(1, EPOCHS + 1):
        print(f"Training {i}/{EPOCHS}")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=experiment_name, callback=evaluate_enz_callback, progress_bar=True)
        model.save(model_save_path)

    print("------------ Training Finished ! ------------")

if __name__ == "__main__":
    scrs = np.array([0.002, 0.00244444, 0.00288889, 0.00333333, 0.00377778,
                    0.00422222, 0.00466667, 0.00511111, 0.00555556, 0.006])

    mueopts = np.array([0.10444444, 0.11333333, 0.12222222, 0.13111111, 0.14])
       #0.10444444, 0.11333333, 0.12222222, 0.13111111, 0.14      ])


    # Create the meshgrid
    SCR_grid, MUE_grid = np.meshgrid(scrs, mueopts)

    # Flatten for easy iteration
    SCR_flat = SCR_grid.ravel()
    MUE_flat = MUE_grid.ravel()

    # Create a list to store processes
    processes = []

    # Create and start a new process for each configuration
    for i,j in zip(SCR_flat,MUE_flat):
        experiment_name = f"scratch_scr-{i}_mopt-{j}"
        p = multiprocessing.Process(target=setup_experiment, args=(experiment_name, 'SAC', i, j))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("All experiments completed!")
