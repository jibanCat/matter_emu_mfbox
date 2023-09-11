"""
Data loader for the matter power spectrum
"""

import os

import numpy as np
from scipy.interpolate import interp1d

from .latin_hypercube import map_to_unit_cube_list


def interpolate(log10_k: np.ndarray, Y: np.ndarray, log10_ks: np.ndarray) -> np.ndarray:
    """
    interpolate the log P(k) based on a given log10 ks

    Parameters:
    ---
    log10_k: k bins of the input function.
    Y: output of the input function.
    log10_ks: the interpolant we want to interpolate the input function on.
    """

    # select interpolatable bins at LR
    ind = (log10_ks <= log10_k.max()) * (log10_ks >= log10_k.min())

    # currently the truncation is done at outer loop,
    # the ind here is just for checking length for interpolant.
    # log10_ks = log10_ks[ind]

    new_kbins = np.sum(ind)
    assert new_kbins > 0
    print(new_kbins, log10_k.shape[0], Y.shape[-1])
    assert new_kbins == log10_ks.shape[0]

    # original LR powerspectrum shape
    n_parms, kbins = Y.shape

    # initialize new Y array for interpolated LR powerspecs
    Y_new = np.full((n_parms, new_kbins), np.nan)

    # loop over each power spec: 1) each LH set; 2) each redshift bin
    for i in range(n_parms):
        f = interp1d(log10_k, Y[i, :])

        Y_new[i, :] = f(log10_ks)

        # remove the interpolant
        del f

    print(
        "[Info] rebin powerspecs from {} k bins to {} k bins.".format(kbins, new_kbins)
    )

    return Y_new


def interp_lf_to_hf_bins(
    folder_lf: str,
    folder_hf: str,
    folder_test_hf: str,
    output_folder: str = "50_dmonly64_3_fullphysics512_mpgadget",
):
    """
    Prepare training data, for power spectra with different k bins between high- and low-fidelity

    we need to
    1) maximum k of LF <- maximum k of HF
    2) minimum k of HF <- minimum k of LF

    The training k range would be confined by low-fidelity's (kmin, kmax).
    """
    # interpolant should be from highres
    log10_ks = np.loadtxt(os.path.join(folder_hf, "kf.txt"))
    Y_hf = np.loadtxt(os.path.join(folder_hf, "output.txt"))
    input_hf = np.loadtxt(os.path.join(folder_hf, "input.txt"))

    log10_ks_test = np.loadtxt(os.path.join(folder_test_hf, "kf.txt"))
    Y_hf_test = np.loadtxt(os.path.join(folder_test_hf, "output.txt"))
    input_hf_test = np.loadtxt(os.path.join(folder_test_hf, "input.txt"))

    assert np.all(log10_ks == log10_ks_test)

    # low-fidelity is in a set of k bins, but we want them
    # to be the same as the high-fidelity k bins
    log10_k = np.loadtxt(os.path.join(folder_lf, "kf.txt"))
    Y_lf = np.loadtxt(os.path.join(folder_lf, "output.txt"))
    input_lf = np.loadtxt(os.path.join(folder_lf, "input.txt"))
    # we want to trim the high-fidelity k to have the same
    # minimum k.
    # highres: log10_ks; lowres: log10_k
    ind_min = (log10_ks >= log10_k.min()) & (log10_ks <= log10_k.max())

    # interpolate: interp(log10_k, Y_lf)(log10_k[ind_min])
    Y_lf_new = interpolate(log10_k, Y_lf, log10_ks[ind_min])

    assert Y_lf_new.shape[1] == log10_ks[ind_min].shape[0]

    # create a folder containing ready-to-use emulator train files
    base_dir = os.path.join("data", output_folder)
    os.makedirs(base_dir, exist_ok=True)

    np.savetxt(os.path.join(base_dir, "train_output_fidelity_0.txt"), Y_lf_new)
    np.savetxt(os.path.join(base_dir, "train_output_fidelity_1.txt"), Y_hf[:, ind_min])
    np.savetxt(os.path.join(base_dir, "test_output.txt"), Y_hf_test[:, ind_min])

    np.savetxt(os.path.join(base_dir, "train_input_fidelity_0.txt"), input_lf)
    np.savetxt(os.path.join(base_dir, "train_input_fidelity_1.txt"), input_hf)
    np.savetxt(os.path.join(base_dir, "test_input.txt"), input_hf_test)

    np.savetxt(os.path.join(base_dir, "kf.txt"), log10_ks[ind_min])

    input_limits = np.loadtxt(os.path.join(folder_lf, "input_limits.txt"))

    np.savetxt(os.path.join(base_dir, "input_limits.txt"), input_limits)


def input_normalize(params: np.ndarray, param_limits: np.ndarray) -> np.ndarray:
    """
    Map the parameters onto a unit cube so that all the variations are
    similar in magnitude.

    :param params: (n_points, n_dims) parameter vectors
    :param param_limits: (n_dim, 2) param_limits is a list
        of parameter limits.
    :return: params_cube, (n_points, n_dims) parameter vectors
        in a unit cube.
    """
    nparams = np.shape(params)[1]
    params_cube = map_to_unit_cube_list(params, param_limits)
    assert params_cube.shape[1] == nparams

    return params_cube


class PowerSpecs:
    """
    A data loader to load multi-fidelity training and test data

    Assume two fidelities.
    """

    def __init__(self, folder: str = "data/50_LR_3_HR/", n_fidelities: int = 2):
        self.n_fidelities = n_fidelities

        # training data
        self.X_train = []
        self.Y_train = []
        for i in range(n_fidelities):
            x_train = np.loadtxt(
                os.path.join(folder, "train_input_fidelity_{}.txt".format(i))
            )
            y_train = np.loadtxt(
                os.path.join(folder, "train_output_fidelity_{}.txt".format(i))
            )

            self.X_train.append(x_train)
            self.Y_train.append(y_train)

        # parameter limits for normalization
        self.parameter_limits = np.loadtxt(os.path.join(folder, "input_limits.txt"))

        # testing data
        self.X_test = []
        self.Y_test = []
        self.X_test.append(np.loadtxt(os.path.join(folder, "test_input.txt")))
        self.Y_test.append(np.loadtxt(os.path.join(folder, "test_output.txt")))

        # load k bins (in log)
        self.kf = np.loadtxt(os.path.join(folder, "kf.txt"))

        assert len(self.kf) == self.Y_test[0].shape[1]
        assert len(self.kf) == self.Y_train[0].shape[1]

    @property
    def X_train_norm(self):
        """
        Normalized input parameters
        """
        x_train_norm = []
        for x_train in self.X_train:
            x_train_norm.append(input_normalize(x_train, self.parameter_limits))

        return x_train_norm

    @property
    def X_test_norm(self):
        """
        Normalized input parameters
        """
        x_test_norm = []
        for x_test in self.X_test:
            x_test_norm.append(input_normalize(x_test, self.parameter_limits))

        return x_test_norm

    @property
    def Y_train_norm(self):
        """
        Normalized training output. Subtract the low-fidelity data with
        their sample mean. Don't change high-fidelity data.
        """
        y_train_norm = []
        for y_train in self.Y_train[:-1]:
            mean = y_train.mean(axis=0)
            y_train_norm.append(y_train - mean)

        # don't change high-fidelity data
        y_train_norm.append(self.Y_train[-1])

        return y_train_norm


class PowerSpecsMFBox:
    """
    A data loader to load multi-fidelity training and test data

    Assume two fidelities.
    """

    def __init__(self, folder: str = "data/50_LR_3_HR/", n_fidelities: int = 2):
        self.n_fidelities = n_fidelities

    def read_from_txt(
        self,
        folder: str = "data/50_LR_3_HR/",
    ):
        """
        Read the multi-fidelity training set from txt files (they have a specific data structure)

        See here https://github.com/jibanCat/matter_multi_fidelity_emu/tree/main/data/50_LR_3_HR
        """
        # training data
        self.X_train = []
        self.Y_train = []

        for i in range(self.n_fidelities):
            x_train = np.loadtxt(
                os.path.join(folder, "train_input_fidelity_{}.txt".format(i))
            )
            y_train = np.loadtxt(
                os.path.join(folder, "train_output_fidelity_{}.txt".format(i))
            )

            self.X_train.append(x_train)
            self.Y_train.append(y_train)

        # parameter limits for normalization
        self.parameter_limits = np.loadtxt(os.path.join(folder, "input_limits.txt"))

        # testing data
        self.X_test = []
        self.Y_test = []
        self.X_test.append(np.loadtxt(os.path.join(folder, "test_input.txt")))
        self.Y_test.append(np.loadtxt(os.path.join(folder, "test_output.txt")))

        # load k bins (in log)
        self.kf = np.loadtxt(os.path.join(folder, "kf.txt"))

        assert len(self.kf) == self.Y_test[0].shape[1]
        assert len(self.kf) == self.Y_train[0].shape[1]

    @property
    def X_train_norm(self):
        """
        Normalized input parameters
        """
        x_train_norm = []
        for x_train in self.X_train:
            x_train_norm.append(input_normalize(x_train, self.parameter_limits))

        return x_train_norm

    @property
    def X_test_norm(self):
        """
        Normalized input parameters
        """
        x_test_norm = []
        for x_test in self.X_test:
            x_test_norm.append(input_normalize(x_test, self.parameter_limits))

        return x_test_norm

    @property
    def Y_train_norm(self):
        """
        Normalized training output. Subtract the low-fidelity data with
        their sample mean. Don't change high-fidelity data.
        """
        y_train_norm = []
        for y_train in self.Y_train[:-1]:
            mean = y_train.mean(axis=0)
            y_train_norm.append(y_train - mean)

        # don't change high-fidelity data
        y_train_norm.append(self.Y_train[-1])

        return y_train_norm


class StellarMassFunctions(PowerSpecs):
    """
    A data loader for stellar mass functions from CAMELS.
    """

    # redefine the init with different loading methods:
    def __init__(
        self,
        folder: str = "data/illustris_smfs/1000_LR_3_HR_test0",
        n_fidelities: int = 2,
    ):
        self.n_fidelities = n_fidelities

        # training data
        self.X_train = []
        self.Y_train = []
        for i in range(n_fidelities):
            x_train = np.loadtxt(
                os.path.join(folder, "train_input_fidelity_{}.txt".format(i))
            )
            y_train = np.loadtxt(
                os.path.join(folder, "train_output_fidelity_{}.txt".format(i))
            )

            self.X_train.append(x_train)
            self.Y_train.append(y_train)

        # parameter limits for normalization
        self.parameter_limits = np.loadtxt(os.path.join(folder, "input_limits.txt"))

        # testing data
        self.X_test = []
        self.Y_test = []
        self.X_test.append(np.loadtxt(os.path.join(folder, "test_input.txt")))
        self.Y_test.append(np.loadtxt(os.path.join(folder, "test_output.txt")))

        if len(self.X_test[0].shape) == 1:
            self.X_test[0] = self.X_test[0][None, :]

        if len(self.Y_test[0].shape) == 1:
            self.Y_test[0] = self.Y_test[0][None, :]

        # Currently no bins for SMFs (TODO: Ask Yongseok)
        # self.kf = np.loadtxt(os.path.join(folder, "kf.txt"))

        # assert len(self.kf) == self.Y_test[0].shape[1]
        # assert len(self.kf) == self.Y_train[0].shape[1]

    @property
    def Y_train_norm(self):
        """
        Normalized training output. Subtract the low-fidelity data with
        their sample mean.
        """
        y_train_norm = []
        for y_train in self.Y_train[:-1]:
            mean = y_train.mean(axis=0)
            y_train_norm.append(y_train - mean)

        # don't change high-fidelity data
        y_train_norm.append(self.Y_train[-1])

        return y_train_norm

    @property
    def Y_test_norm(self):
        """
        Normalized test output. Subtract the low-fidelity data with
        their sample mean.
        """
        y_train_norm = []
        for y_train in self.Y_train[:-1]:
            mean = y_train.mean(axis=0)
            y_train_norm.append(y_train - mean)

        # don't change high-fidelity data
        y_train_norm.append(self.Y_train[-1])

        return y_train_norm
