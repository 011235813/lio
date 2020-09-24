"""Wrapper around Sequential Social Dilemma environment.

This is used for running baseline methods where the new action space is 
{original discrete actions} X continuous reward-giving action.
"""
import numpy as np

from lio.env import ssd


class Env(ssd.Env):

    def __init__(self, config_env):

        super().__init__(config_env)

        self.reward_coeff = self.config.reward_coeff  # cost multiplier
            

