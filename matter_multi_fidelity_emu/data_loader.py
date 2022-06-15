"""
Data loader for the matter power spectrum
"""

import os, json
from typing import List, Optional

import numpy as np
import h5py

from .gpemulator_singlebin import _map_params_to_unit_cube as input_normalize
from .data_loader_dgmgp import interpolate

# put the folder name on top, easier to find and modify
def folder_name(num1: int, res1: int, box1: int, num2: int, res2:int, box2: int, z: float, selected_ind):
    return "Matterpower_{}_res{}box{}_{}_res{}box{}_{}_ind_{}".format(
        num1, res1, box1, num2, res2, box2, "{:.2g}".format(z).replace(".", "_"), "-".join(selected_ind)
    )

def convert_h5_to_txt(
        lf_filename: str = "data/dmo_60_res128box256/cc_emulator_powerspecs.hdf5",
        hf_filename: str = "data/dmo_24_res512box256/cc_emulator_powerspecs.hdf5",
        test_filename: str = "data/dmo_10_res512box256/cc_emulator_powerspecs.hdf5",
        lf_json: str = "data/dmo_60_res128box256/emulator_params.json",
        hf_json: str = "data/dmo_24_res512box256/emulator_params.json",
        test_json: str = "data/dmo_10_res512box256/emulator_params.json",
        hf_selected_ind: Optional[List[int]] = [0, 1, 2], ## these should be the index in the "LF" LHD
    ):
    """
    Convert the h5 files Martin gave me to txt files to be read by the dataloader.

    Keys:
    ----
    bounds : the bounds of the parameter prior 
    powerspecs : Matter power spectrum (rebinned)
    kfmpc : k bins in Mpc/h unit
    params : un-normalized input parameters
    zout : redshifts
    scale_factors : 1 / (zout + 1)

    Note: save each redshift as separate folders
    """

    f_lf = h5py.File(lf_filename, "r")
    f_hf = h5py.File(hf_filename, "r")
    f_test = h5py.File(test_filename, "r")

    with open(lf_json, "r") as param:
        param_lf = json.load(param)
    with open(hf_json, "r") as param:
        param_hf = json.load(param)
    with open(test_json, "r") as param:
        param_test = json.load(param)

    param_limits = np.array(param_lf["bounds"])
    assert np.all(param_limits == np.array(param_hf["bounds"]))
    assert np.all(param_limits == np.array(param_test["bounds"]))

    # make sure all keys are in the file
    keys = ['powerspecs', 'kfmpc', 'params', 'zout']
    for key in keys:
        assert key in f_lf.keys()
        assert key in f_hf.keys()
        assert key in f_test.keys()

    print("Low-fidelity file:")
    print("----")
    print("Resolution:", param_lf["npart"])
    print("Box (Mpc/h):", param_lf["box"])
    print("Shape of redshfits", f_lf["zout"].shape)
    print("Shape of params", f_lf["params"].shape)
    print("Shape of kfmpc", f_lf["kfmpc"].shape)
    print("Shape of powerspecs", f_lf["powerspecs"].shape)
    print("\n")

    # use kfmpc so all redshifts use the same k bins
    kfmpc_lf = f_lf["kfmpc"][()]
    # Different fidelities have different k bins
    # assert np.all(np.abs(kfmpc - f_hf["kfmpc"][()]) < 1e-10)

    zout = f_lf["zout"][()]
    assert np.all( (zout - f_hf["zout"][()]) < 1e-10 )

    # power spectra, all redshifts
    powerspecs_lf = f_lf["powerspecs"][()]

    # input parameters
    x_train_lf = f_lf["params"][()]

    # read power spectrum at ith redshift
    # power spectrum | z = ?
    def get_powerspec_at_z(i: int, powerspecs: np.ndarray) -> np.ndarray:
        return powerspecs[:, i, :]

    # some checking
    last_powerspec = get_powerspec_at_z(len(zout) - 1, powerspecs_lf)
    assert len(last_powerspec) == f_lf["params"].shape[0]
    assert last_powerspec.shape[1] == len(kfmpc_lf)

    first_powerspec = get_powerspec_at_z(0, powerspecs_lf)
    assert len(first_powerspec) == f_lf["params"].shape[0]
    assert first_powerspec.shape[1] == len(kfmpc_lf)


    print("High-fidelity file:")
    print("----")
    print("Resolution:", param_hf["npart"])
    print("Box (Mpc/h):", param_hf["box"])
    print("Shape of redshfits", f_hf["zout"].shape)
    print("Shape of params", f_hf["params"].shape)
    print("Shape of kfmpc", f_hf["kfmpc"].shape)
    print("Shape of powerspecs", f_hf["powerspecs"].shape)
    print("Selected indices:", f_hf["selected_ind"][()])
    print("\n")

    # We only need to save a subset we want to have in the HF training
    selected_ind = f_hf["selected_ind"][()]
    if hf_selected_ind is not None:
        ind_hf_sims = np.isin(
            selected_ind,
            hf_selected_ind,
        )
        print("Check: selected inds are,", selected_ind[ind_hf_sims])
        assert np.all(selected_ind[ind_hf_sims] == hf_selected_ind)
    else:
        ind_hf_sims = np.isin(
            selected_ind,
            np.arange(len(selected_ind)),
        )
        print("Check: selected inds are,", selected_ind[ind_hf_sims])
        assert selected_ind[ind_hf_sims] == hf_selected_ind

    # power spectra, all redshifts
    powerspecs_hf = f_hf["powerspecs"][()]

    # input parameters
    x_train_hf = f_hf["params"][()]

    powerspecs_hf = powerspecs_hf[ind_hf_sims, :, :]
    x_train_hf   = x_train_hf[ind_hf_sims, :]
    print("-> Shape of powerspecs", powerspecs_hf.shape)
    print("-> Selected indices:", selected_ind[ind_hf_sims][()])

    kfmpc_hf = f_hf["kfmpc"][()]
    assert np.all(np.abs(kfmpc_hf - f_test["kfmpc"][()]) < 1e-10)

    # some checking
    last_powerspec = get_powerspec_at_z(len(zout) - 1, powerspecs_hf)
    assert len(last_powerspec) == f_hf["params"][()][ind_hf_sims, :].shape[0]
    assert last_powerspec.shape[1] == len(kfmpc_hf)

    first_powerspec = get_powerspec_at_z(0, powerspecs_hf)
    assert len(first_powerspec) == f_hf["params"][()][ind_hf_sims, :].shape[0]
    assert first_powerspec.shape[1] == len(kfmpc_hf)

    # test files: same resolution as high-fidelity
    print("Test file:")
    print("----")
    print("Resolution:", param_test["npart"])
    print("Box (Mpc/h):", param_test["box"])
    print("Shape of redshfits", f_test["zout"].shape)
    print("Shape of params", f_test["params"].shape)
    print("Shape of kfmpc", f_test["kfmpc"].shape)
    print("Shape of powerspecs", f_test["powerspecs"].shape)
    print("\n")

    # power spectra, all redshifts
    powerspecs_test = f_test["powerspecs"][()]

    # input parameters
    x_train_test = f_test["params"][()]

    # some checking
    last_powerspec = get_powerspec_at_z(len(zout) - 1, powerspecs_test)
    assert len(last_powerspec) == f_test["params"].shape[0]
    assert last_powerspec.shape[1] == len(kfmpc_hf)

    first_powerspec = get_powerspec_at_z(0, powerspecs_test)
    assert len(first_powerspec) == f_test["params"].shape[0]
    assert first_powerspec.shape[1] == len(kfmpc_hf)

    # output training files, one redshift per folder
    for i,z in enumerate(zout):
        print("Preparing training files in {:.3g}".format(z))

        powerspec_lf = get_powerspec_at_z(i, powerspecs_lf)
        powerspec_hf = get_powerspec_at_z(i, powerspecs_hf)
        powerspec_test = get_powerspec_at_z(i, powerspecs_test)

        outdir = folder_name(
            len(x_train_lf),
            param_lf["npart"],
            param_lf["box"],
            len(x_train_hf),
            param_hf["npart"],
            param_hf["box"],
            z,
            selected_ind=hf_selected_ind,
        )

        this_outdir = os.path.join(
                "data",
                "processed",
                outdir,
        )
        os.makedirs(
            this_outdir,
            exist_ok=True,
        )

        # Interpolate the LF, 
        # and limit the HF kmax to the maximum of LF
        # Min k bins LF <= k bins HF <= Max k bins LF
        ind_min = (np.log10(kfmpc_lf).min() <= np.log10(kfmpc_hf)) & (np.log10(kfmpc_hf) <= np.log10(kfmpc_lf).max())

        # interpolate: interp(log10_k, Y_lf)(log10_k[ind_min])
        powerspec_lf_new = interpolate(np.log10(kfmpc_lf), powerspec_lf, np.log10(kfmpc_hf)[ind_min])
        powerspec_hf_new = powerspec_hf[:, ind_min]
        powerspec_test_new = powerspec_test[:, ind_min]
        kfmpc_new = kfmpc_hf[ind_min]


        # new k bins: same k bin over different fidelities
        print("New k bins:")
        print("----")
        print("Shape of kfmpc", kfmpc_new.shape)
        print("Shape of LF powerspecs", powerspec_lf_new.shape)
        print("Shape of HF powerspecs", powerspec_hf_new.shape)
        print("Shape of Test powerspecs", powerspec_test_new.shape)
        print("\n")


        # only power spec needs a loop
        np.savetxt(os.path.join(this_outdir, "train_output_fidelity_0.txt"), powerspec_lf_new)
        np.savetxt(os.path.join(this_outdir, "train_output_fidelity_1.txt"), powerspec_hf_new)
        np.savetxt(os.path.join(this_outdir, "test_output.txt"), powerspec_test_new)

        np.savetxt(os.path.join(this_outdir, "train_input_fidelity_0.txt"), x_train_lf)
        np.savetxt(os.path.join(this_outdir, "train_input_fidelity_1.txt"), x_train_hf)
        np.savetxt(os.path.join(this_outdir, "test_input.txt"), x_train_test)

        np.savetxt(os.path.join(this_outdir, "input_limits.txt"), param_limits)
        np.savetxt(os.path.join(this_outdir, "kf.txt"), np.log10(kfmpc_new))


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
