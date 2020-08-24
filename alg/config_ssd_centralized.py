import sys
sys.path.append('../utils/')

from configdict import ConfigDict


def get_config():

    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.n_episodes = 50000
    config.alg.n_eval = 10
    config.alg.n_test = 3
    config.alg.period = 100

    config.env = ConfigDict()
    # config.env.allow_giving = False
    config.env.asymmetric = False
    config.env.cleaning_penalty = 0.0
    config.env.disable_left_right_action = False
    config.env.disable_rotation_action = True
    # if not None, a fixed global reference frame is used for all agents
    # config.env.global_ref_point = [7, 7]  # for cleanup_wall_15x15
    # config.env.global_ref_point = [3, 7]  # for cleanup_check
    config.env.global_ref_point = [3, 3]  # for cleanup_small
    # config.env.global_ref_point = [2, 2]  # for cleanup_tiny
    # config.env.global_ref_point = None
    config.env.idx_recipient = 0
    config.env.map_name = 'cleanup_small_sym'
    config.env.max_steps = 50
    config.env.n_agents = 2
    config.env.name = 'ssd'
    config.env.obs_cleaned_1hot = False
    # ---------- For 15x15 map ----------
    # config.env.obs_height = 25
    # config.env.obs_width = 25
    # ---------- For 10x10 map ----------
    # config.env.obs_height = 15
    # config.env.obs_width = 15
    # -----------------------------------
    # ---------- for 7x7 map ------------
    config.env.obs_height = 9
    config.env.obs_width = 9
    # -----------------------------------
    # ---------- For 4x5 map ------------
    # config.env.obs_height = 5
    # config.env.obs_width = 5
    # -----------------------------------
    config.env.r_multiplier = 2.0
    config.env.reward_coeff = 1e-4
    config.env.reward_value = 2.0
    config.env.random_orientation = False
    config.env.shuffle_spawn = False
    # config.env.view_size = 2
    config.env.view_size = 4
    # config.env.view_size = 7  # 0.5(height - 1)
    # config.env.view_size = 12
    config.env.cleanup_params = ConfigDict()
    config.env.cleanup_params.appleRespawnProbability = 0.5  # orig 0.05 | 15x15 0.2 | 10x10 0.3 | small 0.5 | min 1.0
    config.env.cleanup_params.thresholdDepletion = 0.6  # orig 0.4 | 15x15 0.4 | 10x10 0.4 | small 0.6 | min 0.02
    config.env.cleanup_params.thresholdRestoration = 0.0  # orig 0 | 15x15 0 | 10x10 0.0 | small 0 | min 0.01
    config.env.cleanup_params.wasteSpawnProbability = 0.5  # orig 0.5 | 15x15 0.5 | 10x10 0.5 | small 0.5 | min 0.5

    config.ia = ConfigDict()
    config.ia.alpha = 0
    config.ia.beta = 0.05
    config.ia.e = 0.95  # what should this be?
    config.ia.enable = True

    config.pg = ConfigDict()
    config.pg.entropy_coeff = 0.1
    config.pg.epsilon_div = 1000
    config.pg.epsilon_end = 0.05
    config.pg.epsilon_start = 0.5
    config.pg.gamma = 0.99
    config.pg.lr_actor = 1e-3
    config.pg.lr_v = 1e-3
    config.pg.reward_type = 'none'
    config.pg.tau = 0.01
    config.pg.use_actor_critic = True

    config.main = ConfigDict()
    config.main.dir_name = 'small_sym_cen'
    config.main.exp_name = 'cleanup'
    config.main.max_to_keep = 10
    config.main.model_name = 'model.ckpt'
    config.main.save_period = 1000000
    config.main.save_threshold = 40
    config.main.seed = 12340
    config.main.summarize = False
    config.main.use_gpu = True

    # difference from Wang et al. 2019: no LSTM layer,
    # first two layers have 64 rather than 32 nodes
    # no value function head
    config.nn = ConfigDict()
    config.nn.kernel = [3, 3]
    config.nn.n_filters = 6
    config.nn.n_h1 = 64
    config.nn.n_h2 = 64
    config.nn.n_h = 128
    config.nn.stride = [1, 1]

    return config
