"""Tests a trained model on SSD."""

import argparse
import os
import random

import numpy as np
import tensorflow as tf

import lio.alg.config_ssd_lio
import lio.alg.config_ssd_pg
import lio.alg.evaluate
import lio.alg.policy_gradient
import lio.env.ssd


def test_lio(config, render=False):

    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    log_path = os.path.join('..', 'results', exp_name, dir_name)
    model_name = config.main.model_name

    n_test = config.alg.n_test

    env = ssd.Env(config.env)

    from lio_ac import LIO

    list_agents = []
    for agent_id in range(env.n_agents):
        list_agents.append(
            LIO(config.lio, env.dim_obs, env.l_action,
                config.nn, 'agent_%d' % agent_id,
                config.env.r_multiplier, env.n_agents,
                agent_id, env.l_action_for_r))

    for agent_id in range(env.n_agents):
        list_agents[agent_id].receive_list_of_agents(list_agents)
        list_agents[agent_id].create_policy_gradient_op()
        list_agents[agent_id].create_update_op()

    for agent_id in range(env.n_agents):
        list_agents[agent_id].create_reward_train_op()

    if config.lio.asymmetric:
        assert config.env.n_agents == 2
        for agent_id in range(env.n_agents):
            list_agents[agent_id].set_can_give(
                agent_id != config.lio.idx_recipient)    

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)

    saver = tf.train.Saver()
    print("Restoring variables from %s" % dir_name)
    saver.restore(sess, os.path.join(log_path, model_name))

    _ = evaluate.test_ssd(n_test, env, sess, list_agents,
                          log=True, log_path=log_path, render=render)


def test_pg(config, render=False):

    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    log_path = os.path.join('..', 'results', exp_name, dir_name)
    model_name = config.main.model_name

    n_test = config.alg.n_test

    reward_type = config.pg.reward_type
    if reward_type == 'continuous':
        if config.pg.use_actor_critic:
            from actor_critic_discrete_continuous import ActorCritic as Alg
        else:
            from policy_gradient_discrete_continuous import PolicyGradient as Alg
    else:
        if config.pg.use_actor_critic:
            from actor_critic import ActorCritic as Alg
        else:
            from policy_gradient import PolicyGradient as Alg    

    env = ssd.Env(config.env)

    list_agents = []
    for agent_id in range(env.n_agents):
        if reward_type == 'continuous':
            list_agents.append(Alg(
                config.pg, env.dim_obs, env.l_action, config.nn,
                'agent_%d' % agent_id, config.env.r_multiplier,
                env.n_agents, agent_id, env.l_action_for_r))
        else:
            if config.ia.enable:
                list_agents.append(Alg(
                    config.pg, env.dim_obs, env.l_action,
                    config.nn, 'agent_%d' % agent_id, agent_id,
                    obs_image_vec=True, l_obs=env.n_agents))
            else:
                list_agents.append(Alg(
                    config.pg, env.dim_obs, env.l_action,
                    config.nn, 'agent_%d' % agent_id, agent_id))

    if config.ia.enable:
        import inequity_aversion
        ia = inequity_aversion.InequityAversion(
            config.ia.alpha, config.ia.beta, config.pg.gamma,
            config.ia.e, config.env.n_agents)
    else:
        ia = None        

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)

    saver = tf.train.Saver()
    print("Restoring variables from %s" % dir_name)
    saver.restore(sess, os.path.join(log_path, model_name))

    _ = evaluate.test_ssd_baseline(n_test, env, sess, list_agents,
                                   render=render, ia=ia)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='lio')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    if args.alg == 'lio':
        config = config_ssd_lio.get_config()
        test_lio(config, args.render)
    elif args.alg == 'pg':
        config = config_ssd_pg.get_config()
        test_pg(config, args.render)
