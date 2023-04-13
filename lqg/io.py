import os
import numpy as np
import scipy.io as spio


# methods for properly reading matlab structs
# taken from https://stackoverflow.com/a/8832212

def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def load_tracking_data(delay=12, clip=120, subtract_mean=True, data_path="data/"):
    """ Load tracking data from Bonnen et al. (2015)

    Args:
        delay: temporal delay between target and response
        clip: clip the first n time steps
        subtract_mean: subtract the mean of both trajectories?

    Returns:
        np.array: data from tracking task
    """

    # factor to convert between pixels and acrmin
    arcscale = 1.32

    # load matlab file
    mat = loadmat(os.path.join(data_path, "data.mat"))

    # get target blob widths
    sigma = (mat["sigma"] * arcscale).round()
    sigmas = np.unique(sigma)

    # convert to numpy.float32
    target = mat["target"].astype(np.float32)
    mouse = mat["response"].astype(np.float32)

    # apply delay and clip the first couple of time steps
    if delay:
        target = target[:, clip:-delay]
        mouse = mouse[:, clip + delay:]
    else:
        target = target[:, clip:]
        mouse = mouse[:, clip:]

    # subtract the mean from both trajectories (in accordance with the analysis by Bonnen et al., 2015)
    # see https://github.com/kbonnen/BonnenEtAl2015_KalmanFilterCode/blob/master/tracking2graphs.m
    if subtract_mean:
        target = target - np.mean(target, axis=1, keepdims=True)
        mouse = mouse - np.mean(mouse, axis=1, keepdims=True)

    # stack data from all trials
    data = np.stack(
        [np.array(
            [target[np.where(sigma == blob_width)[0], :],
             mouse[np.where(sigma == blob_width)[0], :]])
            for blob_width in sigmas])

    # bring in right shape for our analysis methods
    data = data.transpose(0, 2, 3, 1)

    # subtract first time step in x dimension
    data = data - data[:, :, 0, 0][:, :, np.newaxis, np.newaxis]

    return data, sigmas
