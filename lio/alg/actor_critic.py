"""Actor critic with advantage function.

Advantage function is estimated by 1-step TD(0) error.
"""

import numpy as np
import tensorflow as tf

from lio.alg import networks
from lio.utils import util


class ActorCritic(object):

    def __init__(self, config, dim_obs, l_action, nn, agent_name,
                 agent_id=0, obs_image_vec=False, l_obs=None):
        """Initialization.

        Args:
            config: ConfigDict
            dim_obs: list, if obs is an image; or an integer, if obs is a 1D vector
            l_action: integer size of discrete action space
            nn: ConfigDict
            agent_name: string
            agent_id: integer
            obs_image_vec: if true, then agent has both image and 1D vector observation
            l_obs: integer size of 1D vector observation, used only if obs_image_vec
        """
        self.agent_id = agent_id
        self.agent_name = agent_name

        self.dim_obs = dim_obs
        self.image_obs = isinstance(self.dim_obs, list)
        self.l_action = l_action
        self.nn = nn
        # -------------------
        # Used only when agent's observation has both image and 1D vector parts
        self.obs_image_vec = obs_image_vec
        self.l_obs = l_obs
        # -------------------
        
        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.lr_actor = config.lr_actor
        self.lr_v = config.lr_v
        self.tau = config.tau

        self.create_networks()
        self.create_critic_train_op()
        self.create_policy_gradient_op()

    def create_networks(self):
        if self.obs_image_vec:
            self.obs = tf.placeholder(
                tf.float32, [None, self.dim_obs[0], self.dim_obs[1], self.dim_obs[2]],
                'obs')
            self.obs_v = tf.placeholder(tf.float32, [None, self.l_obs], 'obs_v')
            actor_net = networks.actor_image_vec
            value_net = networks.vnet_image_vec
        elif self.image_obs:
            self.obs = tf.placeholder(
                tf.float32, [None, self.dim_obs[0], self.dim_obs[1], self.dim_obs[2]],
                'obs')
            actor_net = networks.actor
            value_net = networks.vnet
        else:
            self.obs = tf.placeholder(tf.float32, [None, self.dim_obs], 'obs')
            actor_net = networks.actor_mlp
            value_net = networks.vnet_mlp
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy_main'):
                with tf.variable_scope('policy'):
                    if self.obs_image_vec:
                        probs = actor_net(self.obs, self.obs_v, self.l_action, self.nn)
                    else:
                        probs = actor_net(self.obs, self.l_action, self.nn)
                self.probs = (1 - self.epsilon) * probs + self.epsilon / self.l_action
                self.log_probs = tf.log(self.probs)
                self.action_samples = tf.multinomial(self.log_probs, 1)

            with tf.variable_scope('v_main'):
                if self.obs_image_vec:
                    self.v = value_net(self.obs, self.obs_v, self.nn)
                else:
                    self.v = value_net(self.obs, self.nn)
            with tf.variable_scope('v_target'):
                if self.obs_image_vec:
                    self.v_target = value_net(self.obs, self.obs_v, self.nn)
                else:
                    self.v_target = value_net(self.obs, self.nn)

        self.policy_params = tf.trainable_variables(
            self.agent_name + '/policy_main/policy')

        self.v_params = tf.trainable_variables(
            self.agent_name + '/v_main')
        self.v_target_params = tf.trainable_variables(
            self.agent_name + '/v_target')
        self.list_initialize_v_ops = []
        self.list_update_v_ops = []
        for idx, var in enumerate(self.v_target_params):
            # target <- main
            self.list_initialize_v_ops.append(
                var.assign(self.v_params[idx]))
            # target <- tau * main + (1-tau) * target
            self.list_update_v_ops.append(
                var.assign(self.tau*self.v_params[idx] + (1-self.tau)*var))

    def run_actor(self, obs, sess, epsilon, obs_v=None):

        feed = {self.obs: np.array([obs]), self.epsilon: epsilon}
        if self.obs_image_vec:
            feed[self.obs_v] = np.array([obs_v])
        action = sess.run(self.action_samples, feed_dict=feed)[0][0]

        return action

    def create_critic_train_op(self):

        self.v_target_next = tf.placeholder(tf.float32, [None], 'v_target_next')
        self.reward = tf.placeholder(tf.float32, [None], 'reward')
        td_target = self.reward + self.gamma * self.v_target_next
        self.loss_v = tf.reduce_mean(tf.square(td_target - tf.squeeze(self.v)))
        self.v_opt = tf.train.AdamOptimizer(self.lr_v)
        self.v_op = self.v_opt.minimize(self.loss_v)

    def create_policy_gradient_op(self):

        self.v_next_ph = tf.placeholder(tf.float32, [None], 'v_next_ph')
        self.v_ph = tf.placeholder(tf.float32, [None], 'v_ph')
        v_td_error = self.reward + self.gamma*self.v_next_ph - self.v_ph

        self.action_taken = tf.placeholder(tf.float32, [None, self.l_action],
                                           'action_taken')
        log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs, self.action_taken), axis=1) + 1e-15)

        self.entropy = -tf.reduce_sum(tf.multiply(self.probs, self.log_probs))

        self.policy_loss = -tf.reduce_sum(
            tf.multiply(log_probs_taken, v_td_error))
        self.loss = self.policy_loss - self.entropy_coeff * self.entropy

        self.policy_grads = tf.gradients(self.loss, self.policy_params)
        grads_and_vars = list(zip(self.policy_grads, self.policy_params))
        self.policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.apply_gradients(grads_and_vars)

    def train(self, sess, buf, epsilon):

        batch_size = len(buf.obs)
        # Update value network
        feed = {self.obs: buf.obs_next}
        if self.obs_image_vec:
            feed[self.obs_v] = buf.obs_v_next
        v_target_next, v_next = sess.run([self.v_target, self.v],
                                         feed_dict=feed)
        v_target_next = np.reshape(v_target_next, [batch_size])
        v_next = np.reshape(v_next, [batch_size])
        feed = {self.obs: buf.obs,
                self.v_target_next: v_target_next,
                self.reward: buf.reward}
        if self.obs_image_vec:
            feed[self.obs_v] = buf.obs_v
        _, v = sess.run([self.v_op, self.v], feed_dict=feed)
        v = np.reshape(v, [batch_size])

        actions_1hot = util.process_actions(buf.action, self.l_action)
        feed = {self.obs: buf.obs,
                self.action_taken: actions_1hot,
                self.reward: buf.reward,
                self.epsilon: epsilon}
        feed[self.v_next_ph] = v_next
        feed[self.v_ph] = v
        if self.obs_image_vec:
            feed[self.obs_v] = buf.obs_v
        _ = sess.run(self.policy_op, feed_dict=feed)

        # Update target network
        sess.run(self.list_update_v_ops)
