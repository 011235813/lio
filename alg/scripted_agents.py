"""Defines agents with hardcoded behaviors.

Used for testing the incentive function during training.
"""

import numpy as np


# Map from name of map to the largest column position
# where a cleaning beam fired from that position can clear waste
cleanup_map_river_boundary = {'cleanup_small_sym': 2,
                              'cleanup_10x10_sym': 3}


cleanup_map_middle = {'cleanup_small_sym': 3,
                      'cleanup_10x10_sym': 4}


class A1(object):
    """An agent who wanders around in the river."""
    def __init__(self, env):

        self.map_name = env.config.map_name

    def run_actor(self, x_pos):
        """Moves to the river, if not already at river region, then moves up and down."""
        if x_pos > cleanup_map_river_boundary[self.map_name]:
            # move left
            return 0
        else:
            # move up or down randomly
            return np.random.choice([2,3])


class A2(object):
    """An agent who goes to the river and keeps firing the cleaning beam."""
    def __init__(self, env):

        self.map_name = env.config.map_name

    def run_actor(self, x_pos):
        if x_pos > cleanup_map_river_boundary[self.map_name]:
            return 0  # move left
        else:
            return 5  # fire cleaning beam


class A3(object):
    """An agent who goes to the middle of map and keeps firing the cleaning beam.

    This agent is not able to clean anything.
    """
    def __init__(self, env):

        self.map_name = env.config.map_name

    def run_actor(self, x_pos):
        if x_pos > cleanup_map_middle[self.map_name]:
            return 0  # move left
        elif x_pos < cleanup_map_middle[self.map_name]:
            return 1  # move right
        else:
            return 5  # fire cleaning beam        


        
