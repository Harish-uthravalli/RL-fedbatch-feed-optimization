from gymnasium.envs.registration import register
import reactor_env


register(
    id="reactor_v2",
    entry_point="reactor_env:Reactor"
)