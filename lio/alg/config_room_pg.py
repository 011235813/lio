"""Experimental parameters for running policy gradient on Escape Room.

Versions supported:
1. without ability to give rewards
2. discrete movement and reward-giving actions
3. discrete movement actions and continuous reward-giving actions
"""

from lio.utils.configdict import ConfigDict


def get_config():

    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.n_episodes = 50000
    config.alg.n_eval = 10
    config.alg.period = 100

    config.env = ConfigDict()
    config.env.max_steps = 5
    config.env.min_at_lever = 1
    config.env.n_agents = 2
    config.env.name = 'er'
    config.env.r_multiplier = 2.0
    config.env.randomize = False
    config.env.reward_sanity_check = False
    config.env.reward_coeff = 1e-4
    config.env.reward_value = 2.0

    config.ia = ConfigDict()
    config.ia.enable = False
    
    config.pg = ConfigDict()
    config.pg.asymmetric = False
    config.pg.centralized = False
    config.pg.entropy_coeff = 0.01
    config.pg.epsilon_div = 1000
    config.pg.epsilon_end = 0.1
    config.pg.epsilon_start = 1.0
    config.pg.gamma = 0.99
    config.pg.idx_recipient = 0
    config.pg.lr_actor = 1e-3
    config.pg.reward_type = 'continuous'  # 'none', 'discrete', 'continuous'
    config.pg.use_actor_critic = False

    config.main = ConfigDict()
    config.main.dir_name = 'er_n2_pg_cont'
    config.main.exp_name = 'er'
    config.main.max_to_keep = 100
    config.main.model_name = 'model.ckpt'
    config.main.save_period = 100000
    config.main.seed = 12340
    config.main.summarize = False
    config.main.use_gpu = False

    config.nn = ConfigDict()
    config.nn.n_h1 = 64
    config.nn.n_h2 = 32
    config.nn.n_hr1 = 64
    config.nn.n_hr2 = 16

    return config
