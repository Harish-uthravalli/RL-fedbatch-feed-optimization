from stable_baselines3.common.callbacks import BaseCallback

class EpisodeCounterCallback(BaseCallback):
    def __init__(self, episode_list, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0  # Tracks the number of episodes
        self.episode_list = episode_list  # List to store the episode counts
    
    def _on_step(self) -> bool:
        # Increment episode count if an episode ends
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if info.get('episode'):  # Gym provides this automatically
                    self.episode_count += 1
        return True

    def _on_training_end(self) -> None:
        # Append the episode count to the list when training ends
        self._append_episode_count()

    def _append_episode_count(self):
        self.episode_list.append(self.episode_count)  # Append to the list
        print(f"Training stopped after {self.episode_count} episodes.")
