"""Gym-compatible version of room_symmetric.py

Used by both LOLA-d (discrete reward-giving actions) and 
LOLA-c (continuous reward-giving actions)
Used by policy gradient with discrete reward-giving actions
Used by policy gradient with no reward-giving
"""

import gym
import numpy as np

from env import room_agent

from gym.spaces import Discrete, Tuple

# from lola.envs.common import OneHot
from utils.common import OneHot


class EscapeRoom(gym.Env):
    """
    A two-agent vectorized environment for symmetric Escape Room game.
    Possible actions for each agent are move to Key, Start, Door
    """
    NAME = 'ER'

    def __init__(self, max_steps, n_agents=2, reward_value=2.0,
                 incentivization_inside_env=True,
                 fixed_episode_length=True, observe_given=True,
                 reward_coeff=1.0):
        """Many repeated variable names to support both LOLA and LIO."""
        self.name = 'er'
        # Only 2 and 3-player are supported 
        assert n_agents == 2 or n_agents == 3
        self.incentivization_inside_env = incentivization_inside_env
        self.max_steps = max_steps
        self.n_agents = n_agents
        # Only support (N=2,M=1) and (N=3,M=2)
        self.min_at_lever = 1 if self.n_agents==2 else 2
        self.NUM_AGENTS = self.n_agents
        self.n_movement_actions = 3
        # giving rewards is simultaneous with movement actions
        self.n_actions = 3 * 2**(self.n_agents-1) if incentivization_inside_env else 3
        self.l_action = self.n_actions
        self.NUM_ACTIONS = self.n_actions
        # give-reward action has this hardcoded value
        self.reward_value = reward_value
        self.fixed_length = fixed_episode_length
        self.observe_given = observe_given
        self.reward_coeff = reward_coeff

        if self.observe_given:
            # self position, others' positions, total given to others
            self.l_obs = 3 + 3*(self.n_agents - 1) + (self.n_agents - 1)
        else:
            # self position, others' positions
            self.l_obs = 3 + 3*(self.n_agents - 1)
        self.NUM_STATES = self.l_obs

        self.action_space = Tuple([Discrete(self.n_actions) for _ in range(self.n_agents)])
        self.observation_space = Tuple([OneHot(self.l_obs) for _ in range(self.n_agents)])
        
        self.actors = [room_agent.Actor(idx, self.n_agents, self.l_obs)
                       for idx in range(self.n_agents)]

        # Map from discrete action to boolean indicator of
        # whether the action involves giving reward
        if self.n_agents == 2:
            self.map_action_to_give = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1}
        elif self.n_agents == 3:
            self.map_action_to_give = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1,
                                       6:1, 7:1, 8:1, 9:2, 10:2, 11:2}

        self.steps = None
        self.solved = False

    def get_door_status(self, actions):
        n_going_to_lever = actions.count(0)
        return n_going_to_lever >= self.min_at_lever

    def calc_reward(self, actions, door_open):
        """
        Args:
        actions: 2-tuple of int
        door_open: Boolean indicator of whether door is open
        """
        assert len(actions) == self.n_agents
        rewards = np.zeros(self.n_agents)
        rewards_env = np.zeros(self.n_agents)  # extrinsic rewards
        list_given = [np.zeros(self.n_agents-1) for _ in range(self.n_agents)]

        if not self.solved:
            for agent_id in range(0, self.n_agents):
                if door_open and (actions[agent_id]%3 == 2):
                    # agent went to an open door
                    rewards[agent_id] += 10
                    rewards_env[agent_id] += 10
                elif (actions[agent_id]%3) == self.actors[agent_id].position:
                    # no penalty for staying at current position
                    pass
                else:
                    rewards[agent_id] += -1
                    rewards_env[agent_id] += -1

                if self.incentivization_inside_env:
                    given = np.zeros(self.n_agents - 1)
                    # Cost for giving reward
                    rewards[agent_id] -= (self.map_action_to_give[actions[agent_id]] *
                                          self.reward_coeff * self.reward_value)
                    # Received rewards
                    if self.n_agents == 2:
                        other_id = 1 - agent_id  # only for the case of 2 agents
                        r = self.map_action_to_give[actions[agent_id]] * self.reward_value
                        rewards[other_id] += r
                        given[0] += r
                    elif self.n_agents == 3:
                        list_others = list(range(0, self.n_agents))
                        del list_others[agent_id]
                        a = actions[agent_id]
                        if (3 <= a and a <= 5) or (9 <= a and a <= 11):
                            rewards[list_others[0]] += self.reward_value
                            given[0] += self.reward_value
                        if (6 <= a and a <= 8) or (9 <= a and a <= 11):
                            rewards[list_others[1]] += self.reward_value
                            given[1] += self.reward_value

                    list_given[agent_id] = given

        return rewards, list_given, rewards_env

    def get_obs(self):
        list_obs = []
        for actor in self.actors:
            list_obs.append(
                actor.get_obs(self.state, self.observe_given))

        return list_obs

    def reset(self):
        self.solved = False
        randomize = (self.n_agents == 3)
        for actor in self.actors:
            actor.reset(randomize)
        self.state = [actor.position for actor in self.actors]
        self.steps = 0
        list_obs = self.get_obs()

        return list_obs

    def step(self, action, list_given_from_outside=None):
        """Take a step in environment.

        Args:
            action: list of integers
            list_given_from_outside: list of np.arrays, used in the case
            of continuous reward-giving actions where incentivization 
            happens in the training script.
        """
        door_open = self.get_door_status(action)
        rewards, list_given, rewards_env = self.calc_reward(action, door_open)
        list_given = (list_given if self.incentivization_inside_env
                      else list_given_from_outside)
        for idx, actor in enumerate(self.actors):
            # check for intentional None
            given = list_given[idx] if list_given else None
            actor.act(action[idx] % 3, given, self.observe_given)
        self.steps += 1
        self.state = [actor.position for actor in self.actors]
        list_obs_next = self.get_obs()

        if door_open and 2 in self.state:
            self.solved = True
            
        if self.fixed_length:
            done = (self.steps == self.max_steps)
        else:
            done = self.solved or (self.steps == self.max_steps)

        info = {'rewards_env': rewards_env}

        return list_obs_next, rewards, done, info
