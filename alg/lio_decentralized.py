"""Fully-decentralized implementation of LIO with opponent modeling.

Currently assumes N=2, to be run on 2-player Escape Room.
"""

import numpy as np
import tensorflow as tf

import lio_agent
import networks

from utils import util


class LIO(lio_agent.LIO):

    def __init__(self, config, l_obs, l_action, nn, agent_name,
                 r_multiplier=2, n_agents=1, agent_id=0,
                 list_agent_id_opp=None):        

        super().__init__(config, l_obs, l_action, nn, agent_name,
                         r_multiplier, n_agents, agent_id)
        self.lr_opp = config.lr_opp
        self.list_agent_id_opp = list_agent_id_opp
        self.policy_new = PolicyNew

    def create_networks(self):
        """Defines placeholders and neural networks.

        Model of the other agent is created here.
        """
        # Placeholders for self
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'l_obs')
        self.action_others = tf.placeholder(
            tf.float32, [None, self.l_action * (self.n_agents - 1)])
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')
        self.action_taken = tf.placeholder(tf.float32, [None, self.l_action],
                                           'action_taken')

        # Placeholders for model of opponent
        self.list_obs_opp = []
        self.list_action_taken_opp = []
        for idx in range(self.n_agents-1):
            self.list_obs_opp.append(
                tf.placeholder(tf.float32, [None, self.l_obs], 'obs_opp_%d'%idx))
            self.list_action_taken_opp.append(
                tf.placeholder(tf.float32, [None, self.l_action], 'action_taken_opp%d'%idx))

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy_main'):
                with tf.variable_scope('policy'):
                    probs = networks.actor_mlp(self.obs, self.l_action, self.nn)
                with tf.variable_scope('eta'):
                    self.reward_function = networks.reward_mlp(self.obs, self.action_others,
                                                               self.nn, n_recipients=self.n_agents)
                self.probs = (1 - self.epsilon) * probs + self.epsilon / self.l_action
                self.log_probs = tf.log(self.probs)
                self.action_samples = tf.multinomial(self.log_probs, 1)

            self.list_logits_opp = []
            self.list_probs_opp = []
            self.list_log_probs_opp = []
            self.list_policy_params_opp = []
            for idx in range(self.n_agents-1):
                with tf.variable_scope('opponent_model_%d'%idx):
                    # Assumes the other agent has same action space and policy network structure
                    logits_opp, probs_opp = networks.actor_mlp(
                        self.list_obs_opp[idx], self.l_action, self.nn, return_logits=True)
                self.list_logits_opp.append(logits_opp)
                probs_opp = (1-self.epsilon)*probs_opp + self.epsilon/self.l_action
                self.list_probs_opp.append(probs_opp)
                self.list_log_probs_opp.append(tf.log(probs_opp))
                self.list_policy_params_opp.append(
                    tf.trainable_variables(self.agent_name + '/opponent_model_%d'%idx))

    def run_actor(self, obs, sess, epsilon, prime=None):
        """Gets action from policy.

        Args:
            obs: observation vector
            sess: TF session
            epsilon: float exploration lower bound
        
        Returns: integer action
        """
        feed = {self.obs: np.array([obs]), self.epsilon: epsilon}
        action = sess.run(self.action_samples, feed_dict=feed)[0][0]

        return action

    def create_opp_modeling_op(self):
        """Defines the op for fitting the opponent policy."""
        
        self.list_opp_op = []
        for idx in range(self.n_agents-1):
            opp_loss = tf.losses.softmax_cross_entropy(
                self.list_action_taken_opp[idx], self.list_logits_opp[idx])
            opp_opt = tf.train.AdamOptimizer(self.lr_opp)
            self.list_opp_op.append(opp_opt.minimize(opp_loss))

    def create_policy_gradient_op(self):
        """Defines the op to be executed on the opponent policy."""

        self.list_r_ext_opp = []
        self.list_ones = []
        self.list_policy_grads_opp = []
        for idx in range(self.n_agents-1):
            # for opponent model update
            r_ext_opp = tf.placeholder(tf.float32, [None], 'r_ext_opp_%d'%idx)
            self.list_r_ext_opp.append(r_ext_opp)
            r_opp = r_ext_opp  # opponent's extrinsic reward

            # Add the reward given to the opponent
            reverse_1hot = 1 - tf.one_hot(indices=self.agent_id, depth=self.n_agents)
            r_opp += self.r_multiplier * tf.reduce_sum(
                tf.multiply(self.reward_function, reverse_1hot), axis=1)

            ones = tf.placeholder(tf.float32, [None], 'ones_%d'%idx)
            self.list_ones.append(ones)
            gamma_prod = tf.math.cumprod(ones * self.gamma)
            returns = tf.reverse(
                tf.math.cumsum(tf.reverse(r_opp * gamma_prod, axis=[0])), axis=[0])
            returns = returns / gamma_prod

            log_probs_taken_opp = tf.log(tf.reduce_sum(tf.multiply(
                self.list_probs_opp[idx], self.list_action_taken_opp[idx]), axis=1) + 1e-15)

            entropy_opp = -tf.reduce_sum(tf.multiply(
                self.list_probs_opp[idx], self.list_log_probs_opp[idx]))

            policy_loss_opp = -tf.reduce_sum(
                tf.multiply(log_probs_taken_opp, returns))
            loss_opp = policy_loss_opp - self.entropy_coeff * entropy_opp

            self.list_policy_grads_opp.append(tf.gradients(
                loss_opp, self.list_policy_params_opp[idx]))

    def create_update_op(self):
        """Defines the op for updating own policy."""
        self.r_ext = tf.placeholder(tf.float32, [None], 'r_ext')
        self.r_from_others = tf.placeholder(tf.float32, [None], 'r_from_others')
        r_total = self.r_ext + self.r_from_others

        self.ones = tf.placeholder(tf.float32, [None], 'ones')
        self.gamma_prod = tf.math.cumprod(self.ones * self.gamma)
        returns_val = tf.reverse(tf.math.cumsum(
            tf.reverse(r_total * self.gamma_prod, axis=[0])), axis=[0])
        returns_val = returns_val / self.gamma_prod

        log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs, self.action_taken), axis=1) + 1e-15)
        entropy = -tf.reduce_sum(
            tf.multiply(self.probs, self.log_probs))
        policy_loss = -tf.reduce_sum(
            tf.multiply(log_probs_taken, returns_val))
        loss = policy_loss - self.entropy_coeff * entropy

        policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = policy_opt.minimize(loss)

    def create_reward_train_op(self):
        """Defines the op for this agent's incentive function."""
        self.returns = tf.placeholder(tf.float32, [None], 'returns')
        self.list_opp_policy_new = []
        list_loss = []
        for idx in range(self.n_agents-1):
            opp_policy_params_new = {}
            for grad, var in zip(self.list_policy_grads_opp[idx],
                                 self.list_policy_params_opp[idx]):
                opp_policy_params_new[var.name] = var - self.lr_actor * grad
            opp_policy_new = self.policy_new(
                opp_policy_params_new, self.l_obs, self.l_action, self.agent_name, idx)
            self.list_opp_policy_new.append(opp_policy_new)

            log_probs_taken = tf.log(
                tf.reduce_sum(tf.multiply(opp_policy_new.probs,
                                          opp_policy_new.action_taken), axis=1))
            loss = -tf.reduce_sum(tf.multiply(log_probs_taken, self.returns))
            list_loss.append(loss)

        # Cost for incentives
        reverse_1hot = 1 - tf.one_hot(indices=self.agent_id, depth=self.n_agents)
        given_each_step = tf.reduce_sum(tf.abs(
            tf.multiply(self.reward_function, reverse_1hot)), axis=1)
        total_given = tf.reduce_sum(tf.multiply(
            given_each_step, self.gamma_prod/self.gamma))

        if self.separate_cost_optimizer:
            self.reward_loss = tf.reduce_sum(list_loss)
        else:
            self.reward_loss = (tf.reduce_sum(list_loss) +
                                self.reg_coeff * total_given)

        if self.optimizer == 'sgd':
            reward_opt = tf.train.GradientDescentOptimizer(self.lr_reward)
            if self.separate_cost_optimizer:
                cost_opt = tf.train.GradientDescentOptimizer(self.lr_cost)
        elif self.optimizer == 'adam':
            reward_opt = tf.train.AdamOptimizer(self.lr_reward)
            if self.separate_cost_optimizer:
                cost_opt = tf.train.AdamOptimizer(self.lr_cost)
        self.reward_op = reward_opt.minimize(self.reward_loss)
        if self.separate_cost_optimizer:
            self.cost_op = cost_opt.minimize(total_given)

    def train_opp_model(self, sess, list_buf, epsilon):
        """Fits opponent model.

        Args:
            sess: TF session
            list_buf: list of all buffers of agents' experiences
            epsilon: float
        """
        for idx, agent_id_opp in enumerate(self.list_agent_id_opp):
            buf = list_buf[agent_id_opp]
            feed = {}
            feed[self.list_obs_opp[idx]] = buf.obs
            feed[self.list_action_taken_opp[idx]] = util.process_actions(
                buf.action, self.l_action)
            _ = sess.run(self.list_opp_op[idx], feed_dict=feed)

    def update(self, sess, buf, epsilon):
        """Training step for own policy.

        Args:
            sess: TF session
            buf: Buffer object
            epsilon: float exploration lower bound
        """
        n_steps = len(buf.obs)
        actions_1hot = util.process_actions(buf.action, self.l_action)
        ones = np.ones(n_steps)
        feed = {self.obs: buf.obs,
                self.action_taken: actions_1hot,
                self.r_ext: buf.reward,
                self.ones: ones,
                self.epsilon: epsilon}

        feed[self.r_from_others] = buf.r_from_others

        _ = sess.run(self.policy_op, feed_dict=feed)

    def train_reward(self, sess, list_buf, list_buf_new, epsilon):
        """Training step for incentive function.

        Args:
            sess: TF session
            list_buf: list of all agents' experience buffers
            list_buf_new: list of all agents' buffers of new experiences, 
                          after policy updates
            epsilon: float exploration lower bound
        """
        buf_self = list_buf[self.agent_id]
        buf_self_new = list_buf_new[self.agent_id]

        n_steps = len(buf_self.obs)
        ones = np.ones(n_steps)

        feed = {}
        feed[self.epsilon] = epsilon
        for idx, agent_id_opp in enumerate(self.list_agent_id_opp):
            buf_other = list_buf[agent_id_opp]
            actions_other_1hot = util.process_actions(buf_other.action, self.l_action)
            feed[self.list_obs_opp[idx]] = buf_other.obs
            feed[self.list_action_taken_opp[idx]] = actions_other_1hot
            feed[self.list_r_ext_opp[idx]] = buf_other.reward
            feed[self.list_ones[idx]] = ones

            buf_other_new = list_buf_new[agent_id_opp]
            actions_other_1hot_new = util.process_actions(buf_other_new.action,
                                                          self.l_action)
            feed[self.list_opp_policy_new[idx].obs] = buf_other_new.obs
            feed[self.list_opp_policy_new[idx].action_taken] = actions_other_1hot_new

        n_steps = len(buf_self_new.obs)
        total_reward = buf_self_new.reward
        returns_new = util.process_rewards(total_reward, self.gamma)
        feed[self.obs] = buf_self.obs
        feed[self.action_others] = util.get_action_others_1hot_batch(
            buf_self.action_all, self.agent_id, self.l_action)
        feed[self.ones] = ones
        feed[self.returns] = returns_new

        if self.separate_cost_optimizer:
            _ = sess.run([self.reward_op, self.cost_op], feed_dict=feed)
        else:
            _ = sess.run(self.reward_op, feed_dict=feed)

        
class PolicyNew(object):
    """New model in which to place the updated model of the other agent."""

    def __init__(self, params, l_obs, l_action, agent_name, opp_idx):
        self.obs = tf.placeholder(tf.float32, [None, l_obs], 'obs_new')
        self.action_taken = tf.placeholder(tf.float32, [None, l_action],
                                           'action_taken')
        prefix = agent_name + '/opponent_model_%d/' % opp_idx
        with tf.variable_scope('policy_new'):
            h1 = tf.nn.relu(
                tf.nn.xw_plus_b(self.obs, params[prefix + 'actor_h1/kernel:0'],
                                params[prefix + 'actor_h1/bias:0']))
            h2 = tf.nn.relu(
                tf.nn.xw_plus_b(h1, params[prefix + 'actor_h2/kernel:0'],
                                params[prefix + 'actor_h2/bias:0']))
            out = tf.nn.xw_plus_b(h2, params[prefix + 'actor_out/kernel:0'],
                                params[prefix + 'actor_out/bias:0'])
        self.probs = tf.nn.softmax(out)    
