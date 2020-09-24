from lio.utils.configdict import ConfigDict


def get_config():

    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.n_episodes = 50000
    config.alg.n_eval = 10
    config.alg.n_test = 2
    config.alg.period = 1000

    config.env = ConfigDict()
    # config.env.allow_giving = False
    config.env.asymmetric = False
    config.env.beam_width = 3
    config.env.cleaning_penalty = 0.0
    config.env.disable_left_right_action = False
    config.env.disable_rotation_action = True
    # if not None, a fixed global reference frame is used for all agents
    # config.env.global_ref_point = [4, 4]  # cleanup_10x10
    # config.env.global_ref_point = [3, 3]  # for cleanup_small
    config.env.global_ref_point = None
    config.env.idx_recipient = 0
    config.env.map_name = 'cleanup_small_sym'
    config.env.max_steps = 50
    config.env.n_agents = 2
    config.env.name = 'ssd'
    config.env.obs_cleaned_1hot = False
    # ---------- For 10x10 map ----------
    # config.env.obs_height = 15
    # config.env.obs_width = 15
    # -----------------------------------
    # ---------- for 7x7 map ------------
    config.env.obs_height = 9
    config.env.obs_width = 9
    # -----------------------------------
    config.env.r_multiplier = 2.0
    config.env.reward_coeff = 1e-4
    config.env.reward_value = 2.0
    config.env.random_orientation = False
    config.env.shuffle_spawn = False
    # config.env.view_size = 5  # for 10x10 map with global ref point
    config.env.view_size = 4
    # config.env.view_size = 7  # 0.5(height - 1)
    config.env.cleanup_params = ConfigDict()
    config.env.cleanup_params.appleRespawnProbability = 0.5  # 10x10 0.3 | small 0.5
    config.env.cleanup_params.thresholdDepletion = 0.6  # 10x10 0.4 | small 0.6
    config.env.cleanup_params.thresholdRestoration = 0.0  # 10x10 0.0 | small 0
    config.env.cleanup_params.wasteSpawnProbability = 0.5  # 10x10 0.5 | small 0.5

    config.ia = ConfigDict()
    config.ia.alpha = 0
    config.ia.beta = 0.05
    config.ia.e = 0.95
    config.ia.enable = False

    config.pg = ConfigDict()
    config.pg.centralized = False
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
    config.main.dir_name = 'small_n2_ac'
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
