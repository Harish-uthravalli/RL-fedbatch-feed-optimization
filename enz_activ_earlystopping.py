from stable_baselines3.common.callbacks import BaseCallback

class EnzymeActivityEarlyStoppingCallback(BaseCallback):
    def __init__(self, patience: int = 10, verbose: int = 1):
        super(EnzymeActivityEarlyStoppingCallback, self).__init__(verbose)
        self.patience = patience  # Number of episodes to wait for improvement
        self.best_enzyme_activity = -float('inf')  # Initialize best enzyme activity as negative infinity
        self.no_improvement_episodes = 0  # Counter for episodes without improvement
    
    def _on_step(self) -> bool:
        """
        Called at each step. This function checks if the current episode is finished
        and evaluates the enzyme activity.
        """
        # Check if an episode has ended
        if self.locals.get('dones', [False])[0]:  # 'dones' indicates the end of an episode
            # Retrieve the last enzyme activity value
            enzyme_activity = self.locals['infos'][0].get('enzyme_activity', 0)
            #print("Current enzyme activity: ",enzyme_activity)
            # Check if enzyme activity improved
            if enzyme_activity > self.best_enzyme_activity:
                self.best_enzyme_activity = enzyme_activity
                self.no_improvement_episodes = 0  # Reset the counter
                if self.verbose > 0:
                    print(f"New best enzyme activity: {self.best_enzyme_activity}")
            else:
                self.no_improvement_episodes += 1
                if self.verbose > 0:
                    print(f"No improvement for {self.no_improvement_episodes} episode(s)")
            
            # Stop training if no improvement over the patience threshold
            if self.no_improvement_episodes >= self.patience:
                print("Stopping early: No improvement in enzyme activity.")
                return False  # Stop training
        
        return True  # Continue training
