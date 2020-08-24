"""Post-processing for inequity aversion agents."""
import numpy as np


class InequityAversion(object):

    def __init__(self, alpha, beta, gamma, e, n_agents):
        """Initialization.

        Args:
            alpha: aversion to disadvantageous inequity
            beta: aversion to advantageous inequity
            gamma: discount factor
            e: lambda in the paper
            n_agents: number of agents
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.e = e
        self.n_agents = n_agents

        self.reset()

    def reset(self):
        # Temporally-smoothed rewards
        self.traces = np.zeros(self.n_agents)

    def compute_rewards(self, rewards_env):
        """Adds inequity aversion terms to each agent's reward.

        Args:
            rewards_env: list of original rewards from environment.
        """
        rewards = np.zeros(self.n_agents)

        # Update temporal smoothed rewards
        for idx, reward in enumerate(rewards_env):
            self.traces[idx] = (self.gamma * self.e * self.traces[idx] +
                               rewards_env[idx])

        # Iterate over agents and compute its inequity aversion reward
        for idx in range(self.n_agents):
            rewards[idx] = rewards_env[idx]
            list_others = list(range(self.n_agents))
            del list_others[idx]

            # Disadvantageous inequity
            alpha_term = 0
            for idx2 in list_others:
                alpha_term += np.maximum(self.traces[idx2] - self.traces[idx], 0)
            rewards[idx] -= self.alpha * alpha_term / (self.n_agents-1)

            # Advantageous inequity
            beta_term = 0
            for idx3 in list_others:
                beta_term += np.maximum(self.traces[idx] - self.traces[idx3], 0)
            rewards[idx] -= self.beta * beta_term / (self.n_agents-1)

        return rewards

    
