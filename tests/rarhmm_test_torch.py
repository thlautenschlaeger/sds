import numpy as np
import torch

from sds_torch.rarhmm import rARHMM
from ssm.hmm import HMM as orgHMM

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(1337)

T = [900, 950]

true_rarhmm = orgHMM(3, 2, observations="ar", transitions="recurrent")

true_z, x = [], []
for t in T:
    _z, _x = true_rarhmm.sample(t)
    true_z.append(_z)
    x.append(torch.from_numpy(_x))

true_ll = true_rarhmm.log_probability([_x.numpy() for _x in x])

# true_rarhmm = rARHMM(nb_states=3, dm_obs=2)
# true_z, x = true_rarhmm.sample(horizon=T)
# true_ll = true_rarhmm.log_probability(x)

my_rarhmm = rARHMM(nb_states=3, dm_obs=2, trans_type='recurrent')
my_rarhmm.initialize(x)
my_ll = my_rarhmm.em(x, nb_iter=100, prec=1e-12)

org_rarhmm = orgHMM(3, 2, observations="ar", transitions="recurrent")
org_ll = org_rarhmm.fit([_x.numpy() for _x in x], method="em")

print("true_ll=", true_ll, "my_ll=", my_ll[-1], "org_ll=", org_ll[-1])
