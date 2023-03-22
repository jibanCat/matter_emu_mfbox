"""
Compute halo mass functions and save into a HDF5 file
"""
from typing import Generator, List, Optional

from .multi_sims import MultiPowerSpec

import os
import numpy as np
import h5py
import re

from .multi_sims import GadgetLoad
from .hmffromfof import HMFFromFOF


class MultiHMF(MultiPowerSpec):

    """
    Output a HDF5 file

    This class tries to convert the HMF data (across redshifts) output from MP-Gadget to a single h5 file.
    This meant to be similar to the routines in sbird.lya_emulator_full.

    Parameters:
    ----
    all_submission_dirs : a List of paths to the MP-Gadget simulation submission folders.
    Latin_json : path to the Latin hypercube json file.
    selected_ind : select a subset of LHD to generate the h5 file.
    """

    def __init__(
        self,
        all_submission_dirs: List[str],
        Latin_json: str,
        selected_ind: Optional[np.ndarray],
        scale_factors: List[float] = [1.0000, 0.8333, 0.6667, 0.5000, 0.3333, 0.2500],
    ) -> None:
        super().__init__(
            all_submission_dirs, Latin_json=Latin_json, selected_ind=selected_ind
        )

        # assign attrs for loading MPGadget power specs
        self.scale_factors = scale_factors

        self.selected_ind = selected_ind

    def create_hdf5(self, hdf5_name: str = "MultiHMF.hdf5") -> None:

        """
        - Create a HDF5 file for hmfs from multiple simulations.
        - Parameters from Latin HyperCube sampling stored in first layer,
        the order of the sampling is the same as the order of simulations.
        - HMFs stored as a large array in the first layer.
        - Stores redshift and scale factor arrays
        - Store k bins.
        TODO: add a method to append new simulations to a created hdf5.
        <KeysViewHDF5 ['flux_vectors', 'kfkms', 'kfmpc', 'params', 'zout']>

        """
        # open a hdf5 file to store simulations
        with h5py.File(hdf5_name, "w") as f:
            # Save the selected ind for comparing with LF
            if self.selected_ind is not None:
                f.create_dataset("selected_ind", data=np.array(self.selected_ind))

            # store the sampling from Latin Hyper cube dict into datasets:
            # since the sampling size should be arbitrary, we should use
            # datasets instead of attrs to stores these sampling arrays
            for key, val in self.Latin_dict.items():
                f.create_dataset(key, data=val)

            # Make input parameter column
            parameter_names = self.Latin_dict["parameter_names"]
            # (num of sims, num of parameters)
            params = np.full(
                (len(self.Latin_dict[parameter_names[0]]), len(parameter_names),),
                fill_value=np.nan,
            )
            for ith_param, pname in enumerate(parameter_names):
                params[:, ith_param] = self.Latin_dict[pname]
            assert np.sum(np.isnan(params)) < 1
            f.create_dataset("params", data=params)

            f.create_dataset("scale_factors", data=np.array(self.scale_factors))
            f.create_dataset("zout", data=1 / np.array(self.scale_factors) - 1)

            # Get the shape first
            # TODO: might be a better way append arrays
            ps_test = MPGadgetPowerSpec(
                self.all_submission_dirs[0], self.scale_factors[0]
            )  # Note: there are default rebinning parameters

            kk = ps_test.k0  # Note: assumption is all k bins are the same
            ps_size = ps_test.powerspecs.shape[0]
            mode_size = ps_test.modes.shape[0]
            assert kk.shape[0] == ps_size
            assert mode_size == ps_size

            # (num of simulations, num of redshifts, num of k bins)
            sim_powerspecs = np.full(
                (len(self.all_submission_dirs), len(self.scale_factors), ps_size,),
                fill_value=np.nan,
            )
            sim_modes = np.full(
                (len(self.all_submission_dirs), len(self.scale_factors), ps_size,),
                fill_value=np.nan,
            )

            f.create_dataset("kfmpc", data=np.array(kk))

            for ith_sim, submission_dir in enumerate(self.all_submission_dirs):
                for jth_red, scale_factor in enumerate(self.scale_factors):
                    ps = MPGadgetPowerSpec(submission_dir, scale_factor)

                    sim_powerspecs[ith_sim, jth_red, :] = ps.powerspecs
                    sim_modes[ith_sim, jth_red, :] = ps.modes

            assert np.sum(np.isnan(sim_powerspecs)) < 1
            f.create_dataset("powerspecs", data=sim_powerspecs)
            f.create_dataset("modes", data=sim_modes)

    @staticmethod
    def load_PowerSpecs(
        all_submission_dirs: List[str], scale_factor: float
    ) -> Generator:
        """
        Iteratively load the PowerSpec class
        """
        for submission_dir in all_submission_dirs:
            yield MPGadgetPowerSpec(submission_dir, scale_factor)
