"""
Data loader for the matter power spectrum
"""

import os
from typing import List

import numpy as np
import h5py

from .gpemulator_singlebin import _map_params_to_unit_cube as input_normalize


# put the folder name on top, easier to find and modify
def folder_name(num1: int, res1: int, box1: int, num2: int, res2:int, box2: int, z: float):
    return "Matterpower_{}_res{}box{}_{}_res{}box{}_{}".format(
        num1, res1, box1, num2, res2, box2, "{:.2g}".format(z).replace(".", "_")
    )

def convert_h5_to_txt(
        lf_filename: str = "data/emu_smallbox_lowres/cc_emulator_flux_vectors_tau1000000.hdf5",
        hf_filename: str = "data/emu_smallbox_highres/cc_emulator_flux_vectors_tau1000000.hdf5",
        test_filename: str = "data/emu_testset/cc_emulator_flux_vectors_tau1000000.hdf5",
        lf_json: str = "data/emu_smallbox_lowres/emulator_params.json",
        hf_json: str = "data/emu_smallbox_highres/emulator_params.json",
        test_json: str = "data/emu_testset/emulator_params.json",
    ):
    """
    Convert the h5 files Martin gave me to txt files to be read by the dataloader.

    Keys:
    ----
    flux_vectors : Lyman alpha flux power spectrum
    kfkms : k bins in km/s unit
    kfmpc : k bins in Mpc/h unit
    params : un-normalized input parameters
    zout : redshifts

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

    param_limits = np.array(param_lf["param_limits"])
    assert np.all(param_limits == np.array(param_hf["param_limits"]))
    assert np.all(param_limits == np.array(param_test["param_limits"]))

    # make sure all keys are in the file
    # TODO: also save kfkms for plotting purpose
    keys = ['flux_vectors', 'kfkms', 'kfmpc', 'params', 'zout']
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
    print("Shape of kfkms", f_lf["kfkms"].shape)
    print("Shape of kfmpc", f_lf["kfmpc"].shape)
    print("Shape of flux vectors", f_lf["flux_vectors"].shape)
    print("\n")

    # use kfmpc so all redshifts use the same k bins
    kfmpc = f_lf["kfmpc"][()]
    assert np.all(np.abs(kfmpc - f_hf["kfmpc"][()]) < 1e-10)

    zout = f_lf["zout"][()]
    assert np.all( (zout - f_hf["zout"][()]) < 1e-10 )

    # flux power spectra, all redshifts
    flux_vectors_lf = f_lf["flux_vectors"][()]

    # input parameters
    x_train_lf = f_lf["params"][()]

    # read flux power spectrum at ith redshift
    # flux power spectrum | z = ?
    def get_flux_vector_at_z(i: int, flux: np.ndarray) -> np.ndarray:
        return flux[:, i * len(kfmpc) : (i + 1) * len(kfmpc)]

    # some checking
    last_flux_vector = get_flux_vector_at_z(len(zout) - 1, flux_vectors_lf)
    assert len(last_flux_vector) == f_lf["params"].shape[0]
    assert last_flux_vector.shape[1] == len(kfmpc)

    first_flux_vector = get_flux_vector_at_z(0, flux_vectors_lf)
    assert len(first_flux_vector) == f_lf["params"].shape[0]
    assert first_flux_vector.shape[1] == len(kfmpc)


    print("High-fidelity file:")
    print("----")
    print("Resolution:", param_hf["npart"])
    print("Box (Mpc/h):", param_hf["box"])
    print("Shape of redshfits", f_hf["zout"].shape)
    print("Shape of params", f_hf["params"].shape)
    print("Shape of kfkms", f_hf["kfkms"].shape)
    print("Shape of kfmpc", f_hf["kfmpc"].shape)
    print("Shape of flux vectors", f_hf["flux_vectors"].shape)
    print("\n")

    # flux power spectra, all redshifts
    flux_vectors_hf = f_hf["flux_vectors"][()]

    # input parameters
    x_train_hf = f_hf["params"][()]

    # some checking
    last_flux_vector = get_flux_vector_at_z(len(zout) - 1, flux_vectors_hf)
    assert len(last_flux_vector) == f_hf["params"].shape[0]
    assert last_flux_vector.shape[1] == len(kfmpc)

    first_flux_vector = get_flux_vector_at_z(0, flux_vectors_hf)
    assert len(first_flux_vector) == f_hf["params"].shape[0]
    assert first_flux_vector.shape[1] == len(kfmpc)

    # test files: same resolution as high-fidelity
    print("Test file:")
    print("----")
    print("Resolution:", param_test["npart"])
    print("Box (Mpc/h):", param_test["box"])
    print("Shape of redshfits", f_test["zout"].shape)
    print("Shape of params", f_test["params"].shape)
    print("Shape of kfkms", f_test["kfkms"].shape)
    print("Shape of kfmpc", f_test["kfmpc"].shape)
    print("Shape of flux vectors", f_test["flux_vectors"].shape)
    print("\n")

    # flux power spectra, all redshifts
    flux_vectors_test = f_test["flux_vectors"][()]

    # input parameters
    x_train_test = f_test["params"][()]

    # some checking
    last_flux_vector = get_flux_vector_at_z(len(zout) - 1, flux_vectors_test)
    assert len(last_flux_vector) == f_test["params"].shape[0]
    assert last_flux_vector.shape[1] == len(kfmpc)

    first_flux_vector = get_flux_vector_at_z(0, flux_vectors_test)
    assert len(first_flux_vector) == f_test["params"].shape[0]
    assert first_flux_vector.shape[1] == len(kfmpc)

    # output training files, one redshift per folder
    for i,z in enumerate(zout):
        print("Preparing training files in {:.3g}".format(z))

        flux_vector_lf = get_flux_vector_at_z(i, flux_vectors_lf)
        flux_vector_hf = get_flux_vector_at_z(i, flux_vectors_hf)
        flux_vector_test = get_flux_vector_at_z(i, flux_vectors_test)

        outdir = folder_name(
            len(x_train_lf),
            param_lf["npart"],
            param_lf["box"],
            len(x_train_hf),
            param_hf["npart"],
            param_hf["box"],
            z,
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

        # only flux power needs a loop
        np.savetxt(os.path.join(this_outdir, "train_output_fidelity_0.txt"), flux_vector_lf)
        np.savetxt(os.path.join(this_outdir, "train_output_fidelity_1.txt"), flux_vector_hf)
        np.savetxt(os.path.join(this_outdir, "test_output.txt"), flux_vector_test)

        np.savetxt(os.path.join(this_outdir, "train_input_fidelity_0.txt"), x_train_lf)
        np.savetxt(os.path.join(this_outdir, "train_input_fidelity_1.txt"), x_train_hf)
        np.savetxt(os.path.join(this_outdir, "test_input.txt"), x_train_test)

        np.savetxt(os.path.join(this_outdir, "input_limits.txt"), param_limits)
        np.savetxt(os.path.join(this_outdir, "kf.txt"), np.log10(kfmpc))


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
