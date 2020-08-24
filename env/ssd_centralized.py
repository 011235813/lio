"""Interface between multi-agent ssd and centralized control algorithm."""

import numpy as np

from env import ssd

class Env(ssd.Env):

    def __init__(self, config_env):

        super().__init__(config_env)

        # Actual number of agents maintained internally
        self._n_agents = config_env.n_agents
        # Size of individual action space
        self.l_action_base = self.l_action
        # Joint action space is product of individual action space
        self.l_action = self.l_action_base**self._n_agents
        self.n_agents = 1

    def decode_action(self, action):
        """Convert centralized integer to list of individual actions.
        Coding scheme: action = action_n * ... * action_1
        """
        actions = []
        for _ in range(self._n_agents):
            actions.append(int(action % self.l_action_base))
            action = int(action / self.l_action_base)

        return actions

    def step(self, list_action):
        """Takes a step in env.
        
        Args:
            action: [integer] that represents the joint action

        Returns:
            List of observations, reward, done, info
        """
        # Convert to individual actions
        actions = self.decode_action(list_action[0])

        actions = [self.map_to_orig[a] for a in actions]
        actions_dict = {'agent-%d'%idx : actions[idx]
                        for idx in range(self._n_agents)}

        # all objects returned by env.step are dicts
        obs_next, rewards, dones, info = self.env.step(actions_dict)
        self.steps += 1

        obs_next = self.process_obs(obs_next)
        # One global obs is enough
        obs_next = [obs_next[0]]
        rewards = list(rewards.values())
        if self.cleaning_penalty > 0:
            for idx in range(self.n_agents):
                if actions[idx] == 8:
                    rewards[idx] -= self.cleaning_penalty
        # Single team reward
        reward = [np.sum(rewards)]
        # Add up cleaned waste
        info['n_cleaned_each_agent'] = [np.sum(info['n_cleaned_each_agent'])]

        done = dones['__all__'] or self.steps == self.max_steps

        return obs_next, reward, done, info
