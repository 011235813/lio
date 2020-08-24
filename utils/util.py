import numpy as np

def process_actions(actions, l_action):

    n_steps = len(actions)
    actions_1hot = np.zeros([n_steps, l_action], dtype=int)
    actions_1hot[np.arange(n_steps), actions] = 1

    return actions_1hot


def get_action_others_1hot(action_all, agent_id, l_action):
    action_all = list(action_all)
    del action_all[agent_id]
    num_others = len(action_all)
    actions_1hot = np.zeros([num_others, l_action], dtype=int)
    actions_1hot[np.arange(num_others), action_all] = 1

    return actions_1hot.flatten()


def get_action_others_1hot_batch(list_action_all, agent_id, l_action):
    n_steps = len(list_action_all)
    n_agents = len(list_action_all[0])
    matrix = np.stack(list_action_all)  # [n_steps, n_agents]
    self_removed = np.delete(matrix, agent_id, axis=1)
    actions_1hot = np.zeros([n_steps, n_agents-1, l_action], dtype=np.float32)
    grid = np.indices((n_steps, n_agents-1))
    actions_1hot[grid[0], grid[1], self_removed] = 1
    actions_1hot = np.reshape(actions_1hot, [n_steps, l_action*(n_agents-1)])

    return actions_1hot


def process_rewards(rewards, gamma):
    n_steps = len(rewards)
    gamma_prod = np.cumprod(np.ones(n_steps) * gamma)
    returns = np.cumsum((rewards * gamma_prod)[::-1])[::-1]
    returns = returns / gamma_prod

    return returns
    
