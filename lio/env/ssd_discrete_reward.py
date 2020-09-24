"""Wrapper around Sequential Social Dilemma environment.

This is used for running baseline methods (e.g., policy gradient and LOLA)
where the new discrete action space is {original actions} X {reward-giving action}.
"""
import numpy as np

from lio.env import ssd


class Env(ssd.Env):

    def __init__(self, config_env):

        super().__init__(config_env)

        # Allow simultaneous move and give-reward
        # The second half of action range is interpreted as
        # (base action and give-reward-to-the-other-agent)
        # Only support 2-player for now
        assert self.n_agents == 2
        self.reward_coeff = self.config.reward_coeff  # cost multiplier
        self.reward_value = self.config.reward_value
        self.idx_recipient = self.config.idx_recipient if self.config.asymmetric else None
        self.l_action_base = self.l_action
        self.l_action = 2 * self.l_action_base

    def step(self, actions):
        """Takes a step in env.
        
        Args:
            actions: list of integers

        Returns:
            List of observations, list of rewards, done, info
        """
        # The second half of action range is interpreted as
        # simultaneous (base action and give-reward)        
        actions = [self.map_to_orig[a % self.l_action_base] for a in actions]
        actions_dict = {'agent-%d'%idx : actions[idx]
                        for idx in range(self.n_agents)}

        # all objects returned by env.step are dicts
        obs_next, rewards, dones, info = self.env.step(actions_dict)
        self.steps += 1

        obs_next = self.process_obs(obs_next)
        rewards = list(rewards.values())
        if self.cleaning_penalty > 0:
            for idx in range(self.n_agents):
                if actions[idx] == 8:
                    rewards[idx] -= self.cleaning_penalty

        # Store the extrinsic rewards here for use in evaluation,
        # separately from the modifications due to reward-giving actions below
        info['rewards_env'] = np.array(rewards)

        for agent_id in range(self.n_agents):
            if self.config.asymmetric and agent_id == self.idx_recipient:
                continue
            # Agent exercised its reward-giving action
            if actions[agent_id] >= self.l_action_base:
                rewards[agent_id] -= self.reward_coeff * self.reward_value
                # Assumes N=2
                rewards[1-agent_id] += self.reward_value

        done = dones['__all__'] or self.steps == self.max_steps

        return obs_next, rewards, done, info
