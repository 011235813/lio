"""Trains LIO agents on SSD."""
import json
import os
import random
import sys
import time

sys.path.append('../env/')

import numpy as np
import tensorflow as tf

import config_ssd_lio
import evaluate
import ssd

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

    epsilon = config.lio.epsilon_start
    epsilon_step = (
        (epsilon - config.lio.epsilon_end) / config.lio.epsilon_div)

    if isinstance(config.lio.reg_coeff, float):
        reg_coeff = config.lio.reg_coeff
    else:
        reg_coeff = 0.0
        if config.lio.reg_coeff == 'linear':
            reg_coeff_step = 1.0 / n_episodes
        elif config.lio.reg_coeff == 'adaptive':
            # reg_coeff increases from 0 to 1 if every
            # evaluation round has improved performance
            reg_coeff_step = 1.0 / (n_episodes / period)

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
        if config.lio.use_actor_critic:
            list_agents[agent_id].create_critic_train_op()

    for agent_id in range(env.n_agents):
        list_agents[agent_id].create_reward_train_op()

    # This handles the special case of two asymmetric agents,
    # one of which is the reward-giver and the other is the recipient
    if config.lio.asymmetric:
        assert config.env.n_agents == 2
        for agent_id in range(env.n_agents):
            list_agents[agent_id].set_can_give(
                agent_id != config.lio.idx_recipient)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())

    if config.lio.use_actor_critic:
        for agent in list_agents:
            sess.run(agent.list_initialize_v_ops)

    # Verify initialization
    # print('exact')
    # print(sess.run(list_agents[0].v_params[0]))
    # print('model main')
    # print(sess.run(tf.trainable_variables('agent_1/opponent_model_0/v_main')[0]))
    # print('model target')
    # print(sess.run(tf.trainable_variables('agent_1/opponent_model_0/v_target')[0]))
    # input('')

    list_agent_meas = []
    list_suffix = ['given', 'received', 'reward_env',
                   'reward_total', 'waste_cleared',
                   'r_riverside', 'r_beam', 'r_cleared']
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    if isinstance(config.lio.reg_coeff, str):
        header += ',regcoeff'
    header += ',time,reward_env_total\n'
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)

    # Log file for measuring incentive behavior w.r.t. 3 scripted agents
    header = 'episode'
    for idx in range(3):
        header += ',A%d_avg,A%d_stderr' % (idx+1, idx+1)
    header += '\n'
    for idx_replace in [0, 1]:
        with open(os.path.join(log_path, 'measure_%d.csv'%idx_replace), 'w') as f:
            f.write(header)

    # Measure behavior at initialization
    if env.n_agents == 2:
        for idx_replace in [0, 1]:
            evaluate.measure_incentive_behavior(
                env, sess, list_agents, log_path, 0, idx_replace)

    step = 0
    step_train = 0
    idx_episode = 0
    t_start = time.time()
    prev_reward_env = 0
    while idx_episode < n_episodes:

        # print('idx_episode', idx_episode)
        list_buffers = run_episode(sess, env, list_agents, epsilon,
                                   prime=False)
        step += len(list_buffers[0].obs)
        idx_episode += 1

        # Standard learning step for all agents
        for idx, agent in enumerate(list_agents):
            agent.update(sess, list_buffers[idx], epsilon)

        # Run new episode with updated policies
        list_buffers_new = run_episode(sess, env, list_agents,
                                       epsilon, prime=True)
        step += len(list_buffers_new[0].obs)
        idx_episode += 1

        for agent in list_agents:
            if agent.can_give:
                agent.train_reward(sess, list_buffers, list_buffers_new,
                                   epsilon, reg_coeff)

        for agent in list_agents:
            agent.update_main(sess)

        step_train += 1

        if idx_episode % period == 0:
            (given, received, reward_env, reward_total,
             waste_cleared, r_riverside, r_beam,
             r_cleared) = evaluate.test_ssd(n_eval, env, sess, list_agents)

            if config.lio.reg_coeff == 'adaptive':
                performance = np.sum(reward_env)
                sign = 1 if performance > prev_reward_env else -1
                reg_coeff = max(0, min(1.0, reg_coeff + sign*reg_coeff_step))
                prev_reward_env = performance

            combined = np.stack([given, received, reward_env,
                                 reward_total, waste_cleared,
                                 r_riverside, r_beam, r_cleared])
            s = '%d,%d,%d' % (idx_episode, step_train, step)
            for idx in range(env.n_agents):
                s += ','
                s += '{:.2e},{:.2e},{:.2e},{:.2e},{:.2f},{:.2e},{:.2e},{:.2e}'.format(
                    *combined[:, idx])
            reward_env_total = np.sum(combined[2])
            if isinstance(config.lio.reg_coeff, str):
                s += ',%.2e' % reg_coeff
            s += ',%d,%.2e\n' % (int(time.time()-t_start), reward_env_total)
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

            if reward_env_total >= config.main.save_threshold:
                saver.save(sess, os.path.join(log_path, 'model_good_%d'%
                                              idx_episode))

        if env.n_agents == 2 and idx_episode % save_period == 0:
            for idx_replace in [0, 1]:
                evaluate.measure_incentive_behavior(env, sess, list_agents,
                                                    log_path, idx_episode, idx_replace)
            saver.save(sess, os.path.join(log_path, '%s.%d'%(
                model_name, idx_episode)))

        if epsilon > config.lio.epsilon_end:
            epsilon -= epsilon_step

        if config.lio.reg_coeff == 'linear':
            reg_coeff = min(1.0, reg_coeff + reg_coeff_step)

    saver.save(sess, os.path.join(log_path, model_name))


def run_episode(sess, env, list_agents, epsilon, prime=False):

    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    list_obs = env.reset()
    done = False

    budgets = np.zeros(env.n_agents)

    while not done:
        list_actions = []
        list_binary_actions = []
        for agent in list_agents:
            action = agent.run_actor(list_obs[agent.agent_id], sess,
                                     epsilon, prime)
            list_actions.append(action)
            list_binary_actions.append(1 if action == env.cleaning_action_idx else 0)

        list_rewards = []
        total_reward_given_to_each_agent = np.zeros(env.n_agents)
        for agent in list_agents:
            if agent.can_give:
                if env.obs_cleaned_1hot:
                    reward = agent.give_reward(list_obs[agent.agent_id],
                                               list_binary_actions, sess,
                                               budgets[agent.agent_id])
                else:
                    reward = agent.give_reward(list_obs[agent.agent_id],
                                               list_actions, sess,
                                               budgets[agent.agent_id])
            else:
                reward = np.zeros(env.n_agents)
            reward[agent.agent_id] = 0
            total_reward_given_to_each_agent += reward
            list_rewards.append(reward)

        list_obs_next, env_rewards, done, info = env.step(list_actions)
        budgets += env_rewards

        for idx in range(env.n_agents):
            given = np.sum(list_rewards[idx])
            budgets[idx] -= given

        for idx, buf in enumerate(list_buffers):
            buf.add([list_obs[idx], list_actions[idx], env_rewards[idx],
                     list_obs_next[idx], done])
            buf.add_r_from_others(total_reward_given_to_each_agent[idx])
            if env.obs_cleaned_1hot:
                buf.add_action_all(list_binary_actions)
            else:
                buf.add_action_all(list_actions)
            buf.add_budgets(budgets)
            if list_agents[idx].include_cost_in_chain_rule:
                buf.add_r_given(np.sum(list_rewards[idx]))
            
        list_obs = list_obs_next

    return list_buffers


class Buffer(object):

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.obs_next = []
        self.done = []
        self.r_from_others = []
        self.r_given = []
        self.action_all = []
        self.budgets = []

    def add(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.obs_next.append(transition[3])
        self.done.append(transition[4])

    def add_r_from_others(self, r):
        self.r_from_others.append(r)

    def add_action_all(self, list_actions):
        self.action_all.append(list_actions)

    def add_budgets(self, budgets):
        self.budgets.append(budgets)
        
    def add_r_given(self, r):
        self.r_given.append(r)


if __name__ == '__main__':

    config = config_ssd_lio.get_config()
    train_function(config)
