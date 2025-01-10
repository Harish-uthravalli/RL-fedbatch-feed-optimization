import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Define a custom callback to stop training when convergence is achieved
class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience: int = 10, verbose: int = 1):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.patience = patience  # Number of checks before stopping
        self.best_mean_reward = -float('inf')
        self.no_improvement_steps = 0  # Tracks how many steps showed no improvement
    
    def _on_step(self) -> bool:
        # Retrieve training logs from the environment
        if 'episode' in self.locals.get('infos', [{}])[0]:
            episode_rewards = [info['episode']['r'] for info in self.locals['infos'] if 'episode' in info]
            if len(episode_rewards) > 0:
                mean_reward = sum(episode_rewards) / len(episode_rewards)
                
                # Check for improvement
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Mean Reward: {self.best_mean_reward} Improved at timestep: {self.num_timesteps}, No Improvement Counter Resetting after checking: {self.no_improvement_steps} Episodes")
                    self.no_improvement_steps = 0  # Reset the counter
                else:
                    self.no_improvement_steps += 1
                
                
                # Stop training if no improvement over the patience period
                if self.no_improvement_steps >= self.patience:
                    if self.verbose > 0:
                        print(f"Stopping early: No improvement in mean reward. No improvement Counter : {self.no_improvement_steps}")
                        self.no_improvement_steps = 0
                    return False
        
        return True

