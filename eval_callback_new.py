from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os

class EpisodeBasedEvalCallback(BaseCallback):
    def __init__(self, eval_env, model_dir, eval_interval=20, n_eval_episodes=5, patience=20, verbose=1):
        super(EpisodeBasedEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self.patience = patience
        self.episode_count = 0
        self.best_average_enzyme_activity = -float('inf')
        self.no_improvement_count = 0
        self.model_dir = model_dir

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            self.episode_count += 1
            
            if self.episode_count % self.eval_interval == 0:
                avg_enzyme_activity = self._evaluate_model()
                if self.verbose > 0:
                    print(f"-------- Currently at Episode: {self.episode_count} --------")
                    print(f"-------- Average Evaluation Enzyme Activity : {avg_enzyme_activity} --------")
                if avg_enzyme_activity > self.best_average_enzyme_activity:
                    self.best_average_enzyme_activity = avg_enzyme_activity
                    self.no_improvement_count = 0
                    best_model_path = os.path.join(self.model_dir, "best_model.zip")
                    self.model.save(best_model_path)  # Save the best model
                    if self.verbose > 0:
                        print(f"########## New best average enzyme activity: {self.best_average_enzyme_activity} (Model saved) ##########")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        print(f"No improvement for {self.no_improvement_count} evaluations.")

                if self.no_improvement_count >= self.patience:
                    print(f"Early stopping triggered: No improvement for {self.patience} evaluations.")
                    return False
        return True

    def _evaluate_model(self):
        enzyme_activities = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, _, done, _, info = self.eval_env.step(action)
                if done:
                    enzyme_activities.append(obs[1])

        avg_enzyme_activity = np.mean(enzyme_activities)
        return avg_enzyme_activity
