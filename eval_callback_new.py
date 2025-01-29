from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class EpisodeBasedEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_interval=20, n_eval_episodes=5, patience=20, verbose=1):
        """
        Custom callback to evaluate the model after a set number of episodes and stop training if no improvement is seen.

        Args:
            eval_env (gym.Env): The evaluation environment.
            eval_interval (int): The number of episodes after which to evaluate the model.
            n_eval_episodes (int): The number of episodes to run during each evaluation.
            patience (int): Number of evaluations to wait for improvement before stopping training.
            verbose (int): Verbosity level (0: silent, 1: info).
        """
        super(EpisodeBasedEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self.patience = patience
        self.episode_count = 0
        self.best_average_enzyme_activity = -float('inf')  # Track the best average enzyme activity
        self.no_improvement_count = 0  # Counter for early stopping

    def _on_step(self) -> bool:
        """
        Called at each environment step. Tracks the number of completed episodes and evaluates the model at intervals.
        """
        if self.locals["dones"][0]:  # Check if an episode has ended
            self.episode_count += 1
            
            if self.episode_count % self.eval_interval == 0:
                # Perform evaluation
                avg_enzyme_activity = self._evaluate_model()
                if self.verbose > 0:
                    print(f"[Episode {self.episode_count}] Average enzyme activity: {avg_enzyme_activity}")

                # Check for improvement
                if avg_enzyme_activity > self.best_average_enzyme_activity:
                    self.best_average_enzyme_activity = avg_enzyme_activity
                    self.no_improvement_count = 0  # Reset counter
                    if self.verbose > 0:
                        print(f"New best average enzyme activity: {self.best_average_enzyme_activity}")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        print(f"No improvement for {self.no_improvement_count} evaluations.")

                # Stop training if no improvement for 'patience' evaluations
                if self.no_improvement_count >= self.patience:
                    print("Early stopping triggered: No improvement for 20 evaluations.")
                    return False  # Stop training

        return True  # Continue training

    def _evaluate_model(self):
        """
        Run evaluation episodes in the evaluation environment and calculate
        the average enzyme activity.

        Returns:
            float: The average enzyme activity over the evaluation episodes.
        """
        enzyme_activities = []
        steps_per_episode = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_steps = 0
            while not done:
                # Use the trained model to select an action
                action, _ = self.model.predict(obs, deterministic=False)
                obs, _, done, _, info = self.eval_env.step(action)
                
                # Collect enzyme activity from the info dictionary
                #enzyme_activities.append(info.get("enzyme_activity", 0))
                if done:
                    enzyme_activities.append(obs[1])
                    
                    
                episode_steps += 1

            steps_per_episode.append(episode_steps)

        # Calculate the average enzyme activity and steps per episode
        avg_enzyme_activity = np.mean(enzyme_activities)
        avg_steps_per_episode = np.mean(steps_per_episode)

        if self.verbose > 0:
            print(f"Average steps per episode during evaluation: {avg_steps_per_episode}")

        return avg_enzyme_activity
