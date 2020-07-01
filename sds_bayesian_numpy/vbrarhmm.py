from sds_bayesian_numpy.vbarhmm import VBARHMM
from sds_bayesian_numpy.transitions import BayesianNeuralRecurrentTransition

class VBrARHMM(VBARHMM):

    def __init__(self, obs, n_states, act,
                 init_prior, trans_prior, obs_prior, trans_type='neural', trans_kwargs={}):

        super(VBrARHMM, self).__init__(obs, n_states, act,
                                       init_prior, trans_prior, obs_prior)

        self.trans_type = trans_type
        self.trans_model = BayesianNeuralRecurrentTransition(self.n_states, self.obs_dim, self.act_dim, trans_prior,
                                                             **trans_kwargs)
