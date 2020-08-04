from .hmm import HMM
from .arhmm import ARHMM
from .rarhmm import rARHMM
from .erarhmm import erARHMM
from .ensemble import Ensemble

import torch
import os

from gym.envs.registration import register


try:
    register(
        id='HybridPendulumTorch-ID-v1',
        entry_point='sds_torch.envs:HybridPendulumWithCartesianObservation',
        max_episode_steps=1000,
        kwargs={'rarhmm': torch.load(open(os.path.dirname(__file__)
                                          + '/envs/hybrid/models/neural_rarhmm_pendulum_cart.pkl', 'rb'),
                                     map_location='cpu')}
    )
except:
    pass