import numpy as np
from autograd.misc import flatten
from autograd.wrap_util import wraps
from autograd import grad, value_and_grad

from functools import partial

get_name = lambda f: getattr(f, '__name__', '[unknown name]')
get_doc  = lambda f: getattr(f, '__doc__' , '')

def flatten_to_dim(X, d):
    """
    Flatten an array of dimension k + d into an array of dimension 1 + d.

    Example:
        X = npr.rand(10, 5, 2, 2)
        flatten_to_dim(X, 4).shape # (10, 5, 2, 2)
        flatten_to_dim(X, 3).shape # (10, 5, 2, 2)
        flatten_to_dim(X, 2).shape # (50, 2, 2)
        flatten_to_dim(X, 1).shape # (100, 2)

    Parameters
    ----------
    X : array_like
        The array to be flattened.  Must be at least d dimensional

    d : int (> 0)
        The number of dimensions to retain.  All leading dimensions are flattened.

    Returns
    -------
    flat_X : array_like
        The input X flattened into an array dimension d (if X.ndim == d)
        or d+1 (if X.ndim > d)
    """
    assert X.ndim >= d
    assert d > 0
    return np.reshape(X[None, ...], (-1,) + X.shape[-d:])


def series_to_windows(series, window_size):
    num_data = series.shape[0]
    windows = []
    for i in range(0, num_data, window_size):
        windows.append(series[i:i+window_size])

    return windows


def timeseries_to_kernel(series, p):

    num_data = series.shape[0]
    m = series.shape[1]
    lin_prices = series.reshape(-1)
    zs = []
    for t in range(num_data-p + 1):
        start_i = t * m
        end_i = start_i + (p * 1) * m
        zs.append(lin_prices[start_i:end_i])
    Z = np.stack(zs, axis=0)

    return Z


def unflatten_optimizer_step(step):
    """
    Wrap an optimizer step function that operates on flat 1D arrays
    with a version that handles trees of nested containers,
    i.e. (lists/tuples/dicts), with arrays/scalars at the leaves.
    """
    @wraps(step)
    def _step(value_and_grad, x, itr, state=None, *args, **kwargs):
        _x, unflatten = flatten(x)

        def _value_and_grad(x, i):
            v, g = value_and_grad(unflatten(x), i)
            return v, flatten(g)[0]

        _next_x, _next_val, _next_g, _next_state = \
            step(_value_and_grad, _x, itr, state=state, *args, **kwargs)
        return unflatten(_next_x), _next_val, _next_g, _next_state
    return _step


@unflatten_optimizer_step
def adam_step(value_and_grad, x, itr, state=None, step_size=0.001,
              b1=0.9, b2=0.999, eps=10**-8):

    m, v = (np.zeros(len(x)), np.zeros(len(x))) if state is None else state
    val, g = value_and_grad(x, itr)
    m = (1 - b1) * g + b1 * m         # First  moment estimate.
    v = (1 - b2) * (g**2) + b2 * v    # Second moment estimate.
    mhat = m / (1 - b1**(itr + 1))    # Bias correction.
    vhat = v / (1 - b2**(itr + 1))
    x = x - (step_size * mhat) / (np.sqrt(vhat) + eps)
    return x, val, g, (m, v)

@unflatten_optimizer_step
def sgd_step(value_and_grad, x, itr, state=None, step_size=0.001, mass=0.9):
    # Stochastic gradient descent with momentum.
    velocity = state if state is not None else np.zeros(len(x))
    val, g = value_and_grad(x, itr)
    velocity = mass * velocity - (1.0 - mass) * g
    x = x + step_size * velocity
    return x, val, g, velocity

def _generic_sgd(method, loss, x0,  nb_iter=200, state=None, full_output=False):

    step = dict(adam=adam_step, sgd=sgd_step)[method]

    # Initialize outputs
    x, losses, grads = x0, [], []
    for itr in range(nb_iter):
        x, val, g, state = step(value_and_grad(loss), x, itr, state)
        losses.append(val)
        grads.append(g)

    if full_output:
        return x, state
    else:
        return x

adam = partial(_generic_sgd, "adam")

def ensure_args_are_viable_lists(f):
    def wrapper(self, obs, act=None, **kwargs):
        assert obs is not None
        # obs = [np.atleast_2d(obs)] if not isinstance(obs, (list, tuple)) else obs
        obs = [obs] if not isinstance(obs, (list, tuple)) else obs

        if act is None:
            act = []
            for _obs in obs:
                act.append(np.zeros((_obs.shape[0], self.act_dim)))

        act = [np.atleast_2d(act)] if not isinstance(act, (list, tuple)) else act

        return f(self, obs, act, **kwargs)
    return wrapper

def write_data_to_text(path, description:str='test', **kwargs):
    text = description + '\n'
    for key, value in kwargs.items():
        text += "\n" + str(key) + ": " + str(value)
    file = open(path + "/" + description+".txt", "w")

    file.write(text)
