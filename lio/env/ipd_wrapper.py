from lola.envs.prisoners_dilemma import IteratedPrisonersDilemma

class IPD(IteratedPrisonersDilemma):

    def __init__(self, config):

        super().__init__(max_steps=config.max_steps)
        self.n_agents = 2
        self.l_action = 2
        self.l_obs = 5
        self.max_steps = config.max_steps
        self.name = 'ipd'
