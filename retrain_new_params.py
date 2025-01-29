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


scrs = np.linspace(0.002, 0.006, 10)
mue_opts = np.linspace(0.06, 0.14, 10)

[X, Y] = np.meshgrid(scrs, mue_opts)

Z = np.zeros_like(X)
counter = 1
for a in range(X.shape[0]):
    for b in range(X.shape[1]):
        print(f"---------- index: {a}, {b} ----------")
        # Training loop
        scr = X[a, b] 
        mue_opt = Y[a, b]
        print(f"-------- Running {counter}/100 ----------")
        print(f"------ SCR_OPT : {scr}, MUE_MAX : {mue_opt } ------")
        
        register(
            id="reactor_v2",
            entry_point="reactor_env:Reactor",
            kwargs={'experiment_name':"default"}
        )

        # Experiment Details
        experiment_name = f"sac_rt_scr-{round(scr,4)}_mopt-{round(mue_opt,3)}"
        #experiment_name= "testing_eval"
        # Model Specifications
        model_name = config.MODEL

        # Logfiles and model save path
        trained_model_name = 'sac_2'
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

        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=1, min_evals=5, verbose=1)
        
        enz_epi_eval_callback = EpisodeBasedEvalCallback(env_test, eval_interval=100, n_eval_episodes=10, patience=20, verbose=1)
        
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
                callback=[enz_epi_eval_callback, episode_counter], 
                progress_bar=True
            )
            model.save(model_save_path)

        print("Episode counts during training:")
        print(episode_counts)
        Z[a, b] = episode_counts[0]
        print(f"Index running at {a}, {b}")

        counter+= 1


import pandas as pd
import matplotlib.pyplot as plt

# Flatten X, Y, Z for DataFrame
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()

# Create a DataFrame
df = pd.DataFrame({
    'SCR': X_flat,
    'MUE_OPT': Y_flat,
    'EPISODE_COUNT': Z_flat
})

# Save DataFrame to CSV
csv_filename = "scr_mue_episode_counts.csv"
df.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}")

# Plot the data as a heatmap
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label="Episode Count")
plt.xlabel('SCR')
plt.ylabel('MUE_OPT')
plt.title('Episode Counts Heatmap')
plot_filename = "scr_mue_episode_counts_heatmap.png"
plt.savefig(plot_filename)
plt.show()

print(f"Heatmap saved to {plot_filename}")
