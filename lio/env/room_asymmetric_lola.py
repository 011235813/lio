"""Gym-compatible version of room_asymmetric.py for LOLA."""

import gym
import numpy as np

import room_agent

from gym.spaces import Discrete, Tuple

from lola.envs.common import OneHot


class EscapeRoomAsym(gym.Env):
    """
    A two-agent vectorized environment for asymmetric Escape Room game.
    Possible actions for A1 are move to Start, Door
    Possible actions for A2 are move to Start, Lever
    """
    NAME = 'ER'

    def __init__(self, max_steps, allow_giving=True):

        self.allow_giving = allow_giving
        self.max_steps = max_steps
        self.n_agents = 2
        self.NUM_AGENTS = self.n_agents
        # giving rewards is simultaneous with movement actions
        self.n_actions_1 = 4 if allow_giving else 2
        self.n_actions_2 = 2
        self.n_actions = [self.n_actions_1, self.n_actions_2]
        # give-reward action has this hardcoded value
        self.reward_value = 2.0

        # A1 observes: self position, A2's pos, A2's chosen action, total amount given
        self.l_obs_1 = 2 + 2 + 2 + 1
        # A2 observes self pos
        self.l_obs_2 = 2
        self.l_obs = [self.l_obs_1, self.l_obs_2]

        self.action_space = \
            Tuple([Discrete(self.n_actions_1), Discrete(self.n_actions_2)])
        self.observation_space = \
            Tuple([OneHot(self.l_obs_1), OneHot(self.l_obs_2)])
        
        # Map from discrete action to boolean indicator of
        # whether the action involves giving reward
        self.map_action_to_give = {0:0, 1:0, 2:1, 3:1}

        self.step_count = None
        self.solved = False

    def get_door_status(self, actions):
        """Door is open if Agent 2 goes to the lever."""
        return actions[1] == 1

    def calc_reward(self, actions, door_open):
        """
        Args:
        actions: 2-tuple of int
        door_open: Boolean indicator of whether door is open
        """
        assert len(actions) == self.n_agents
        rewards = np.zeros(self.n_agents)

        if not self.solved:
            # A1 reward
            if door_open and actions[0]%2 == 1:
                # A1 went to an open door
                rewards[0] = 10
            else:
                # Otherwise, penalty at each time step
                rewards[0] = -1

            # A2 reward
            if actions[1] != self.a2_pos:
                # penalty for moving away from current position
                rewards[1] = -1

            given_reward = self.map_action_to_give[actions[0]] * self.reward_value
            # Cost for giving reward
            rewards[0] -= given_reward
            # Received rewards
            rewards[1] += given_reward
            self.a1_total_given += given_reward

        return rewards

    def get_obs(self):
        list_obs = []
        # Deliberately use the base obs size for A1 here. Action of A2 will be
        # concatenated to A1's base obs in the training loop.
        obs_1 = np.zeros(self.l_obs_1 - 1 - 2)
        obs_1[self.a1_pos] = 1
        obs_1[2 + self.a2_pos] = 1
        list_obs.append(obs_1)
        obs_2 = np.zeros(self.l_obs_2)
        obs_2[self.a2_pos] = 1
        list_obs.append(obs_2)

        return list_obs

    def reset(self):
        self.solved = False
        self.a1_pos = 0
        self.a2_pos = 0
        self.a1_total_given = 0
        self.step_count = 0
        list_obs = self.get_obs()

        return list_obs

    def step(self, action):
        door_open = self.get_door_status(action)
        rewards = self.calc_reward(action, door_open)
        self.a1_pos = action[0] % 2
        self.a2_pos = action[1]
        self.step_count += 1
        list_obs_next = self.get_obs()
        if door_open and self.a1_pos == 1:
            self.solved = True
        done = (self.step_count == self.max_steps)

        return list_obs_next, rewards, done    
