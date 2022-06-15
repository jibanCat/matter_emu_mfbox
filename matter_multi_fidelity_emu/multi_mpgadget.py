"""
Loading the total mass power spectrum from default
MP-Gadget simulations.
"""
from SimulationRunner.multi_sims import MultiPowerSpec, PowerSpec
from typing import Tuple, List, Optional, Generator

import os
import re
from glob import glob

import numpy as np
from scipy.interpolate import interp1d
import h5py

from .rebin_powerspectrum import modecount_rebin

powerspec_fn = lambda scale_factor: "powerspectrum-{:.4f}.txt".format(scale_factor)

def load_total_mass(submission_dir: str, scale_factor: int, minmodes: int = 200, ndesired: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the total mass power spectrum from MP-Gadget powerspectrum files

    Parameters:
    ----
    submission_dir: your MP-Gadget simulation submission folder.
    scale_factor: the scale factor ( 1 / (1 + z0)) of the power spectrum.

    Returns:
    ----
    kk: k bins of the power spectrum (after rebinning)
    pk: power spectrum
    modes: number of modes at each bin
    """

    load_fn = lambda f: os.path.join(submission_dir, "output", f)

    # check if the simulation run until z=0
    assert os.path.exists(load_fn(powerspec_fn(1)))

    # store all available scale factors
    powerspec_files = glob(
            os.path.join(submission_dir, "output", "powerspectrum-*.txt")
    )
    # store the length first
    length = len(powerspec_files)
    # these files should be in-order, so must be super careful
    # here search a list of scale factor first by a regex
    regex = load_fn("powerspectrum-(.*).txt")
    # those powerspec files are named by the scale factors from 0.1 to 1
    scale_factors = [get_number(regex, f) for f in powerspec_files]
    # make it in order so that we can read in in-order
    scale_factors = sorted(scale_factors)
    assert scale_factors[-1] == 1.0  # make sure you reach the z = 0
    assert length == len(scale_factors)

    # make sure the input scale factor is available
    diff = np.abs(np.array(scale_factors) - scale_factor)
    if min(diff) > 0.0001:
        raise Exception("input scale factor {} is not in the MP-Gadget output files.".format(scale_factor))
    filename = "powerspectrum-{:.4f}.txt".format(scale_factors[np.argmin(diff)])
    print("[Info] Loading powerspetrum file ...", filename)

    # load the file
    powerspec = np.loadtxt(load_fn(filename))
    kk, pk, modes, _ = powerspec.T
    print("[Info] number of k bins before rebinning", kk.shape)

    # rebin powerspectrum
    kk, pk, modes = modecount_rebin(kk, pk, modes, minmodes=minmodes, ndesired=ndesired)

    return kk, pk, modes

def get_number(regex: str, filename: str) -> float:
    """
    Get the number out of the filename using regex
    """
    r = re.compile(regex)

    out = r.findall(filename)

    assert len(out) == 1
    del r

    return float(out[0])

class MPGadgetPowerSpec(PowerSpec):

    """
    Loading MP-Gadget powerspec.

    Loading all snapshots for MP-Gadget outputs.

    Note: the training set will be directly outputed from MultiMPGadgetPowerSpec
    """

    def __init__(self,
        submission_dir: str = "test/", z0 : float = 0.0,
    ) -> None:
        super(PowerSpec, self).__init__(submission_dir)

        # read into arrays
        # Matter power specs from simulations
        kk, pk, modes = self.read_powerspec(submission_dir=submission_dir, z0=z0)

        self._scale_factors = np.array([1 / (1 + z0)])

        self._k0 = kk
        self._powerspecs = pk
        self._modes = modes # TODO: implement uncertainty

        # Matter power specs from CAMB linear theory code
        redshifts, out = self.read_camblinear(self.camb_files)

        self._camb_redshifts = redshifts
        self._camb_matters = out


    @property
    def powerspecs(self) -> np.ndarray:
        """
        P(k) from output folder. The same length as k0.
        """
        return self._powerspecs

    @property
    def modes(self) -> np.ndarray:
        """
        Number of modes counted in a given k bin. The same length as k0.
        """
        return self._modes

    @property
    def poisson_noise(self) -> np.ndarray:
        """
        Poisson noise,
        Var = number counts
        here return the square root of the variance
        """
        return np.sqrt(self._modes)

    @property
    def k0(self) -> np.ndarray:
        """
        k from output folder. The same length as powerspecs
        """
        return self._k0

    def read_powerspec(self, submission_dir: str, z0: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read power spectrum from a output folder

        Parameters:
        ----
        z0 : the redshift of the power spectrum you want to load.
        """
        tol = 1e-4 # tolerance
        # the scale factor you condition on
        scale_factor = 1 / (1 + z0)

        # the maximum k is controlled by Ng
        kk, pk, modes = load_total_mass(submission_dir, scale_factor=scale_factor)

        # filter out NaN values
        ind = ~np.isnan(kk)
        assert np.all(ind == ~np.isnan(kk))

        return kk, pk, modes

class MultiMPGadgetPowerSpec(MultiPowerSpec):

    """
    Output a HDF5 file

    Interpolate the LowRes (if necessary).
    """

    def __init__(self, all_submission_dirs: List[str], Latin_json: str, selected_ind: Optional[np.ndarray],
        z0 : float = 0.0,) -> None:
        super().__init__(all_submission_dirs, Latin_json=Latin_json, selected_ind=selected_ind)

        # assign attrs for loading MPGadget power specs
        self.z0 = z0

    def create_hdf5(self, hdf5_name: str = "MultiMPGadgetPowerSpec.hdf5") -> None:

        """
        - Create a HDF5 file for powerspecs from multiple simulations.
        - Each simulation stored in subgroup, includeing powerspecs and
        camb linear power specs.
        - Each subgroup has their own simulation parameters extracted from
        SimulationICs.json to reproduce this simulation.
        - Parameters from Latin HyperCube sampling stored in upper group level,
        the order of the sampling is the same as the order of simulations.

        TODO: add a method to append new simulations to a created hdf5.
        """
        # open a hdf5 file to store simulations
        with h5py.File(hdf5_name, "w") as f:
            # store the sampling from Latin Hyper cube dict into datasets:
            # since the sampling size should be arbitrary, we should use
            # datasets instead of attrs to stores these sampling arrays
            for key, val in self.Latin_dict.items():
                f.create_dataset(key, data=val)

            # using generator to iterate through simulations,
            # PowerSpec stores big arrays so we don't want to load
            # everything to memory
            for i, ps in enumerate(self.load_PowerSpecs(self.all_submission_dirs)):
                sim = f.create_group("simulation_{}".format(i))

                # store arrays to sim subgroup
                # Simulation Power spectra:
                sim.create_dataset("scale_factors", data=np.array(ps.scale_factors))
                sim.create_dataset("powerspecs", data=ps.powerspecs)
                sim.create_dataset("k0", data=ps.k0)

                # for calculating the uncertainty
                sim.create_dataset("modes", data=ps.modes)
                sim.create_dataset("poisson_noise", data=ps.poisson_noise)

                sim.create_dataset("camb_redshifts", data=np.array(ps.camb_redshifts))
                sim.create_dataset("camb_matters", data=ps.camb_matters)

                # stores param json to metadata attrs
                for key, val in ps.param_dict.items():
                    sim.attrs[key] = val

    @staticmethod
    def load_PowerSpecs(all_submission_dirs: List[str]) -> Generator:
        """
        Iteratively load the PowerSpec class
        """
        for submission_dir in all_submission_dirs:
            yield MPGadgetPowerSpec(submission_dir)
