import numpy as np
import gymnasium as gym

MAIN_ENGINE_ACTION = 2
SIDE_ENGINE_ACTIONS = (1, 3)
MAIN_ENGINE_COST = 0.30
SIDE_ENGINE_COST = 0.03
CRASH_REWARD = -100.0
LANDING_REWARD = 100.0


class ChannelDecomposedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._prev = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev = self._shaping_components(obs)
        return obs, info

    @staticmethod
    def _shaping_components(obs):
        x, y, vx, vy, angle, vang, leg1, leg2 = obs
        return {
            "pos": -100 * np.sqrt(x ** 2 + y ** 2),
            "vel": -100 * np.sqrt(vx ** 2 + vy ** 2),
            "ang": -100 * abs(angle),
            "legs": 10 * (leg1 + leg2),
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cur = self._shaping_components(obs)

        terminal_override = terminated and abs(reward) >= 50.0
        if terminal_override:
            ch_a, ch_b, ch_c = 0.0, float(reward), 0.0
        else:
            ch_a = (
                (cur["pos"] - self._prev["pos"])
                + (cur["vel"] - self._prev["vel"])
                + (cur["ang"] - self._prev["ang"])
            )
            ch_b = cur["legs"] - self._prev["legs"]
            m_power = 1.0 if action == MAIN_ENGINE_ACTION else 0.0
            s_power = 1.0 if action in SIDE_ENGINE_ACTIONS else 0.0
            ch_c = -(m_power * MAIN_ENGINE_COST + s_power * SIDE_ENGINE_COST)
            residual = reward - (ch_a + ch_b + ch_c)
            ch_a += residual

        self._prev = cur
        info = dict(info)
        info["reward_channels"] = {"A": ch_a, "B": ch_b, "C": ch_c}
        info["reward_original"] = float(reward)
        return obs, reward, terminated, truncated, info
