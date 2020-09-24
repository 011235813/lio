"""Policy gradient with discrete and continuous actions."""

import numpy as np
import tensorflow as tf

from scipy.special import expit

import lio.alg.networks
import lio.alg.policy_gradient
from lio.utils import util


class PolicyGradient(policy_gradient.PolicyGradient):

    def __init__(self, config, dim_obs, l_action, nn, agent_name,
                 r_multiplier, n_agents=1, agent_id=0, l_action_for_r=None):
        """
        Args:
            config: configDict
            dim_obs: either scalar or 3-tuple (for image observation)
            l_action: int
            nn: configDict containing neural network sizes
            agent_name: string
            r_multiplier: positive float
            n_agents: int
            agent_id: int
            l_action_for_r: size of action input to reward function
        """
        self.n_agents = n_agents
        self.r_multiplier = r_multiplier
        self.l_action_for_r = l_action_for_r if l_action_for_r else l_action
        super().__init__(config, dim_obs, l_action, nn, agent_name, agent_id)

    def create_networks(self):
        """Initialize placeholders, networks, and outputs."""

        if self.image_obs:
            self.obs = tf.placeholder(
                tf.float32, [None, self.dim_obs[0], self.dim_obs[1], self.dim_obs[2]],
                'obs')
            actor_net = networks.actor
            reward_net = networks.reward
        else:
            self.obs = tf.placeholder(tf.float32, [None, self.dim_obs], 'obs')
            actor_net = networks.actor_mlp
            reward_net = networks.reward_mlp
        self.action_others = tf.placeholder(
            tf.float32, [None, self.l_action_for_r*(self.n_agents - 1)],
            'action_others')
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy_main'):
                with tf.variable_scope('policy'):
                    probs = actor_net(self.obs, self.l_action, self.nn)
                self.probs = (1 - self.epsilon) * probs + self.epsilon / self.l_action
                self.log_probs = tf.log(self.probs)
                self.action_samples = tf.multinomial(self.log_probs, 1)

                with tf.variable_scope('reward'):
                    reward_mean = reward_net(self.obs, self.action_others,
                                             self.nn, self.n_agents,
                                             output_nonlinearity=None)
                stddev = tf.ones_like(reward_mean)
                # Will be squashed by sigmoid later
                self.reward_dist = tf.distributions.Normal(
                    loc=reward_mean, scale=stddev)
                self.reward_sample = self.reward_dist.sample()
                
    def give_reward(self, obs, action_all, sess):
        """Computes reward to give to all agents.

        Args:
            obs: TF tensor
            action_all: list of movement action indices
            sess: TF session

        Returns both the sample from Gaussian and the value 
        after passing through sigmoid. The former is needed 
        to compute log probs during training.
        """
        action_others_1hot = util.get_action_others_1hot(
            action_all, self.agent_id, self.l_action_for_r)
        feed = {self.obs: np.array([obs]),
                self.action_others: np.array([action_others_1hot])}
        reward_sample = sess.run(self.reward_sample, feed_dict=feed).flatten()
        reward = self.r_multiplier * expit(reward_sample)  # sigmoid

        return reward, reward_sample        

    def create_policy_gradient_op(self):
        """Policy gradient op.

        Assumes pi(movement, give reward|s) = pi(movement|s)pi(give reward|s)
        """
        self.r_ext = tf.placeholder(tf.float32, [None], 'r_ext')
        self.ones = tf.placeholder(tf.float32, [None], 'ones')
        self.gamma_prod = tf.math.cumprod(self.ones * self.gamma)
        returns = tf.reverse(tf.math.cumsum(
            tf.reverse(self.r_ext * self.gamma_prod, axis=[0])), axis=[0])
        returns = returns / self.gamma_prod

        # movement part
        self.action_taken = tf.placeholder(tf.float32, [None, self.l_action],
                                           'action_taken')
        self.log_probs_move = tf.log(tf.reduce_sum(
            tf.multiply(self.probs, self.action_taken), axis=1) + 1e-15)

        # reward part
        self.r_sampled = tf.placeholder(tf.float32, [None, self.n_agents],
                                        'r_sampled')
        self.log_probs_reward = self.reward_dist.log_prob(self.r_sampled)
        # Account for the change of variables due to passing through sigmoid and scaling
        # The formula implemented here is p(y) = p(x) |det dx/dy | = p(x) |det 1/(dy/dx)|
        sigmoid_derivative = tf.math.sigmoid(self.r_sampled) * (
            1 - tf.math.sigmoid(self.r_sampled)) / self.r_multiplier
        self.log_probs_reward = (tf.reduce_sum(self.log_probs_reward, axis=1) -
                                 tf.reduce_sum(tf.math.log(sigmoid_derivative), axis=1))

        # Assume pi(a,r|s) = pi(a|s)*pi(r|s)
        self.log_probs_total = self.log_probs_move + self.log_probs_reward

        self.entropy = -tf.reduce_sum(tf.multiply(self.probs, self.log_probs))

        self.policy_loss = -tf.reduce_sum(
            tf.multiply(self.log_probs_total, returns))
        self.loss = self.policy_loss - self.entropy_coeff * self.entropy

        self.policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.minimize(self.loss)

    def train(self, sess, buf, epsilon):
        """On-policy training step.

        Args:
            sess: TF session
            buf: Buffer object
            epsilon: float
        """
        n_steps = len(buf.obs)
        actions_1hot = util.process_actions(buf.action, self.l_action)
        ones = np.ones(n_steps)
        feed = {self.obs: buf.obs,
                self.action_taken: actions_1hot,
                self.r_sampled: buf.r_sampled,
                self.r_ext: buf.reward,
                self.ones: ones,
                self.epsilon: epsilon}
        feed[self.action_others] = util.get_action_others_1hot_batch(
            buf.action_all, self.agent_id, self.l_action_for_r)

        _ = sess.run(self.policy_op, feed_dict=feed)
