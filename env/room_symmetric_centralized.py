"""Escape room for running a centralized baseline method."""
import numpy as np


class EscapeRoom(object):

    def __init__(self, config):

        self.name = 'er'
        # Internal value for convenience
        self._n_agents = config.n_agents
        # Only 2 and 3-player are supported 
        assert self._n_agents == 2 or self._n_agents == 3
        self.max_steps = config.max_steps
        # Only support (N=2,M=1) and (N=3,M=2)
        self.min_at_lever = config.min_at_lever
        self.randomize = config.randomize

        self.l_action = 3
        # All agents' positions
        self.l_obs = 3*self._n_agents

        self.steps = None
        self.solved = False
        self.state = None

        # Size of individual action space
        self.l_action_base = self.l_action
        # Joint action space is product of individual action space
        # Outward facing value for centralized alg
        self.l_action = self.l_action_base**self._n_agents
        self.n_agents = 1  
        # self._n_agents = n_agents

    def get_door_status(self, actions):
        n_going_to_lever = actions.count(0)
        return n_going_to_lever >= self.min_at_lever

    def calc_reward(self, actions, door_open):
        """
        Args:
        actions: tuple of int
        door_open: Boolean indicator of whether door is open
        """
        # rewards = np.zeros(self.n_agents)
        reward = 0
        if not self.solved:
            for idx, action in enumerate(actions):
                if door_open and action == 2:
                    reward += 10
                elif action == self.state[idx]:
                    pass  # no penalty for staying at current position
                else:
                    reward += -1

        return reward

    def get_obs(self):
        """Returns multi-hot representation of self.state.

        e.g. [1,0,2] --> [0,1,0,1,0,0,0,0,1]
        """
        obs = np.zeros(self.l_obs)
        for idx in range(self._n_agents):
            obs[3*idx + self.state[idx]] = 1
        return obs

    def decode_action(self, action):
        """Convert centralized integer to list of individual actions.
        Coding scheme: action = action_3 * action_2 * action_1
        """
        actions = []
        for _ in range(self._n_agents):
            actions.append(int(action % self.l_action_base))
            action = int(action / self.l_action_base)

        return actions

    def step(self, list_action):
        """Takes a step in env.
        
        Args:
            list_action: [integer] that represents the joint action

        Returns:
            List of observations, reward, done, info
        """
        actions = self.decode_action(list_action[0])
        door_open = self.get_door_status(actions)
        reward = self.calc_reward(actions, door_open)
        reward = np.array([reward])

        for idx in range(self._n_agents):
            self.state[idx] = actions[idx]
        self.steps += 1
        list_obs_next = [self.get_obs()]

        if door_open and 2 in self.state:
            self.solved = True
        done = ((door_open and 2 in self.state) or
                self.steps == self.max_steps)
        info = {'rewards_env': reward}

        return list_obs_next, reward, done, info        

    def reset(self):
        self.solved = False
        self.state = (np.random.randint(3, size=self._n_agents) if
                      self.randomize else np.ones(self._n_agents, dtype=int))
        self.steps = 0
        obs = self.get_obs()

        return [obs]
