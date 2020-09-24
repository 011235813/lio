"""Trains independent policy gradient or actor-critic agents.

Supported environments are symmetric Escape Room and SSD

Supports four baselines:
1. without ability to give rewards
2. discrete movement and reward-giving actions
3. discrete movement actions and continuous reward-giving actions
4. inequity aversion agents
"""
import argparse
import json
import os
import random
import time

import numpy as np
import tensorflow as tf

import config_room_pg
import config_ssd_pg
import evaluate


def train_function(config):

    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    log_path = os.path.join('..', 'results', exp_name, dir_name)
    model_name = config.main.model_name
    save_period = config.main.save_period

    os.makedirs(log_path, exist_ok=True)

    # Keep a record of parameters used for this run
    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

    n_episodes = config.alg.n_episodes
    n_eval = config.alg.n_eval
    period = config.alg.period

    reward_type = config.pg.reward_type
    assert reward_type in ['continuous', 'discrete', 'none']
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

    epsilon = config.pg.epsilon_start
    epsilon_step = (
        (epsilon - config.pg.epsilon_end) / config.pg.epsilon_div)

    # ------------------ Initialize env----------------------#
    if config.env.name == 'er':
        if reward_type == 'continuous':
            from env import room_symmetric
            env = room_symmetric.Env(config.env)
        else:  # 'discrete' reward-giving actions or no reward
            if 'centralized' in config.pg and config.pg.centralized:
                from env import room_symmetric_centralized as room
                env = room.EscapeRoom(config.env)
            else:
                from env import room_symmetric_baseline as room
                allow_giving = (reward_type == 'discrete')
                observe_given = (reward_type == 'discrete')
                env = room.EscapeRoom(                
                    config.env.max_steps, config.env.n_agents,
                    config.env.reward_value,
                    incentivization_inside_env=allow_giving,
                    fixed_episode_length=False,
                    observe_given=observe_given,
                    reward_coeff=config.env.reward_coeff)
        dim_obs = env.l_obs
        l_action_for_r = None
    elif config.env.name == 'ssd':
        if reward_type == 'none':
            if 'centralized' in config.pg and config.pg.centralized:
                from env import ssd_centralized
                env = ssd_centralized.Env(config.env)
            else:
                from env import ssd
                env = ssd.Env(config.env)
        elif reward_type == 'continuous':
            from env import ssd_continuous_reward
            env = ssd_continuous_reward.Env(config.env)
        else:
            from env import ssd_discrete_reward
            env = ssd_discrete_reward.Env(config.env)
        dim_obs = env.dim_obs
        l_action_for_r = env.l_action_for_r
    # --------------------------------------------------------#

    # ----------------- Initialize agents ---------------- #
    list_agents = []
    for agent_id in range(env.n_agents):
        if reward_type == 'continuous':
            list_agents.append(Alg(
                config.pg, dim_obs, env.l_action, config.nn,
                'agent_%d' % agent_id, config.env.r_multiplier,
                env.n_agents, agent_id, l_action_for_r))
        else:
            if config.ia.enable:
                list_agents.append(Alg(
                    config.pg, dim_obs, env.l_action,
                    config.nn, 'agent_%d' % agent_id, agent_id,
                    obs_image_vec=True, l_obs=env.n_agents))
            else:
                list_agents.append(Alg(
                    config.pg, dim_obs, env.l_action,
                    config.nn, 'agent_%d' % agent_id, agent_id))
    # ------------------------------------------------------- #

    if config.ia.enable:
        import inequity_aversion
        ia = inequity_aversion.InequityAversion(
            config.ia.alpha, config.ia.beta, config.pg.gamma,
            config.ia.e, config.env.n_agents)
    else:
        ia = None

    config_proto = tf.ConfigProto()
    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())

    if config.pg.use_actor_critic:
        for agent in list_agents:
            sess.run(agent.list_initialize_v_ops)

    list_agent_meas = []
    if config.env.name == 'er':
        if reward_type == 'continuous':
            list_suffix = ['reward_total', 'n_lever', 'n_door',
                           'received', 'given']
        else:
            list_suffix = ['reward_total', 'n_lever', 'n_door']
    elif config.env.name == 'ssd':
        if reward_type == 'none':
            list_suffix = ['reward_env', 'waste_cleared']
        else:
            list_suffix = ['given', 'received', 'reward_env',
                           'reward_total', 'waste_cleared',
                           'r_riverside', 'r_beam', 'r_cleared']            
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    if config.env.name == 'er':
        header += ',steps_per_eps\n'
    else:
        header += ',time,reward_env_total\n'
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)

    step = 0
    step_train = 0
    t_start = time.time()
    for idx_episode in range(1, n_episodes + 1):

        list_buffers = run_episode(sess, env, list_agents, epsilon,
                                   reward_type, ia)
        step += len(list_buffers[0].obs)

        for idx, agent in enumerate(list_agents):
            agent.train(sess, list_buffers[idx], epsilon)

        step_train += 1

        if idx_episode % period == 0:

            if config.env.name == 'ssd':
                if reward_type == 'none':
                    (reward_env, waste_cleared) = evaluate.test_ssd_baseline(
                        n_eval, env, sess, list_agents, ia=ia)
                    combined = np.stack([reward_env, waste_cleared])
                else:
                    if reward_type == 'discrete':
                        (given, received, reward_env, reward_total,
                         waste_cleared, r_riverside, r_beam,
                         r_cleared) = evaluate.test_ssd_baseline(
                             n_eval, env, sess, list_agents,
                             allow_giving=True)
                    else:  # continuous
                        (given, received, reward_env, reward_total,
                         waste_cleared, r_riverside, r_beam,
                         r_cleared) = evaluate.test_ssd(
                             n_eval, env, sess, list_agents,
                             alg='ac')
                    combined = np.stack([given, received, reward_env,
                                         reward_total, waste_cleared,
                                         r_riverside, r_beam, r_cleared])
            elif config.env.name == 'er':
                if reward_type == 'continuous':
                    (reward_total, n_move_lever, n_move_door,
                     rewards_received, rewards_given,
                     steps_per_episode, r_lever,
                     r_start, r_door) = evaluate.test_room_symmetric(
                         n_eval, env, sess, list_agents, alg='pg')
                    combined = np.stack([reward_total, n_move_lever,
                                         n_move_door, rewards_received,
                                         rewards_given])
                else:
                    (reward_total, n_lever, n_door,
                     steps_per_episode) = evaluate.test_room_symmetric_baseline(
                         n_eval, env, sess, list_agents)
                    combined = np.stack([reward_total, n_lever,
                                         n_door])
            s = '%d,%d,%d' % (idx_episode, step_train, step)
            for idx in range(env.n_agents):
                s += ','
                if config.env.name == 'ssd':
                    if reward_type == 'none':
                        s += '{:.3e},{:.2f}'.format(*combined[:, idx])
                    else:
                        s += ('{:.2e},{:.2e},{:.2e},{:.2e},{:.2f}'
                              ',{:.2e},{:.2e},{:.2e}').format(*combined[:, idx])
                elif config.env.name == 'er':
                    if reward_type == 'continuous':
                        s += '{:.3e},{:.3e},{:.3e},{:.3e},{:.3e}'.format(
                            *combined[:, idx])
                    else:
                        s += '{:.3e},{:.3e},{:.3e}'.format(
                            *combined[:, idx])

            if config.env.name == 'ssd':
                reward_env_total = (np.sum(combined[0]) if reward_type == 'none'
                                    else np.sum(combined[2]))
                s += ',%d,%.2e\n' % (int(time.time()-t_start),
                                     reward_env_total)
            elif config.env.name == 'er':
                s += ',%.2f\n' % steps_per_episode
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

            if (config.env.name == 'ssd' and
                reward_env_total >= config.main.save_threshold):
                saver.save(sess, os.path.join(log_path, 'model_good_%d'%
                                              idx_episode))

        if idx_episode % save_period == 0:
            saver.save(sess, os.path.join(log_path, '%s.%d'%(
                model_name, idx_episode)))

        if epsilon > config.pg.epsilon_end:
            epsilon -= epsilon_step

    saver.save(sess, os.path.join(log_path, model_name))
    

def run_episode(sess, env, list_agents, epsilon, reward_type, ia=None):
    """Runs one episode and returns experiences

    Args:
        enable_ia: if True, computes inequity aversion rewards and 
                   agents have extra observation vector
        ia: InequityAversion object
    """
    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    list_obs = env.reset()
    done = False
    if ia:
        ia.reset()
        # all agents observe the same vector of smoothed rewards
        obs_v = np.array(ia.traces)  

    while not done:
        list_actions = []
        list_binary_actions = []
        for agent in list_agents:
            if ia:
                action = agent.run_actor(list_obs[agent.agent_id],
                                         sess, epsilon, obs_v)
            else:
                action = agent.run_actor(list_obs[agent.agent_id],
                                         sess, epsilon)
            list_actions.append(action)
            if env.name == 'ssd' and reward_type == 'continuous':
                list_binary_actions.append(1 if action == env.cleaning_action_idx else 0)

        if reward_type == 'continuous':
            list_rewards = []
            list_r_sampled = []
            total_reward_given_to_each_agent = np.zeros(env.n_agents)
            total_reward_given_by_each_agent = np.zeros(env.n_agents)
            for idx, agent in enumerate(list_agents):
                if env.name == 'er':
                    reward, r_sampled = agent.give_reward(
                        list_obs[agent.agent_id], list_actions, sess)
                else:  # ssd
                    if env.obs_cleaned_1hot:
                        reward, r_sampled = agent.give_reward(
                            list_obs[agent.agent_id], list_binary_actions, sess)
                    else:
                        reward, r_sampled = agent.give_reward(
                            list_obs[agent.agent_id], list_actions, sess)
                reward[agent.agent_id] = 0
                total_reward_given_to_each_agent += reward
                total_reward_given_by_each_agent[idx] = np.sum(reward)
                reward = np.delete(reward, agent.agent_id)
                list_rewards.append(reward)
                list_r_sampled.append(r_sampled)
        
        if env.name == 'ssd':
            (list_obs_next, env_rewards, done, info) = env.step(list_actions)
            if ia:
                env_rewards = ia.compute_rewards(env_rewards)
                obs_v_next = np.array(ia.traces)
        elif env.name == 'er':
            if reward_type == 'continuous':
                (list_obs_next, env_rewards,
                 done) = env.step(list_actions, list_rewards)
            else:
                (list_obs_next, env_rewards,
                 done, _) = env.step(list_actions)

        if reward_type == 'continuous':
            # For continuous reward-giving actions, this is handled here.
            # For discrete actions, it is handled in the env.
            for idx in range(env.n_agents):
                # add rewards received from others
                env_rewards[idx] += total_reward_given_to_each_agent[idx]
                # subtract rewards given to others
                if env.name == 'ssd':
                    # cost may be reduced for fair comparison
                    env_rewards[idx] -= (env.reward_coeff * total_reward_given_by_each_agent[idx])
                else:
                    env_rewards[idx] -= total_reward_given_by_each_agent[idx]

        for idx, buf in enumerate(list_buffers):
            buf.add([list_obs[idx], list_actions[idx],
                     env_rewards[idx], list_obs_next[idx], done])
            if reward_type == 'continuous':
                buf.add_r_sampled(list_r_sampled[idx])
                if env.name == 'ssd' and env.obs_cleaned_1hot:
                    buf.add_action_all(list_binary_actions)
                else:
                    buf.add_action_all(list_actions)
            if ia:
                buf.add_obs_v(obs_v, obs_v_next)
            
        list_obs = list_obs_next
        if ia:
            obs_v = obs_v_next

    return list_buffers


class Buffer(object):

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.obs = []
        self.obs_v = []
        self.action = []
        self.action_all = []
        self.r_sampled = []
        self.reward = []
        self.obs_next = []
        self.obs_v_next = []
        self.done = []

    def add(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.obs_next.append(transition[3])
        self.done.append(transition[4])
        
    def add_r_sampled(self, r_sampled):
        self.r_sampled.append(r_sampled)

    def add_action_all(self, list_actions):
        self.action_all.append(list_actions)

    def add_obs_v(self, obs_v, obs_v_next):
        self.obs_v.append(obs_v)
        self.obs_v_next.append(obs_v_next)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str,
                        choices=['er', 'ssd'])
    args = parser.parse_args()    

    if args.exp == 'er':
        config = config_room_pg.get_config()
    elif args.exp == 'ssd':
        config = config_ssd_pg.get_config()

    train_function(config)
