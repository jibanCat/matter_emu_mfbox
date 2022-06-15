"""
Here provide an example data pipeline for extract power spectra from MP-Gadget simulations
(which means the pipeline might be different for users using other simulations.)
"""
from typing import List, Optional
import os, json

from matter_multi_fidelity_emu.multi_mpgadget import MultiMPGadgetPowerSpec


# This is my naming convention, might not be true for everyone.
# Note: you could define your own naming function in the argument.
fn_dmonly_outdir = lambda i, npart, box: "test-{}-{}-dmonly_{}".format(npart, box, str(i).zfill(4))

# put the folder name on top, easier to find and modify
def h5_folder_name(num: int, npart: int, box: int,):
    return "dmo_{}_res{}box{}".format(
        num, npart, box,
    )

def example_h5(
        npart: int,
        box: int,
        num_simulations: int,
        scale_factors : List[float] = [1.0000, 0.8333, 0.6667, 0.5000, 0.3333, 0.2500],
        optimal_ind: Optional[List] = [  ],
        Latin_json: str = "data_slhd/dmo_128_512_60/matterSLHD_60.json",
        fn_outdir = fn_dmonly_outdir,
        base_dir: str = "data_slhd/dmo_128_512_60/",
        hdf5_name: str = "cc_emulator_powerspecs.hdf5",
    ):
    """
    Generate training data for the selected points from dmonly simulations.
    """

    # This is just creating a list of filenames
    test_dir_fn = lambda i: os.path.join(base_dir, fn_outdir(i, npart, box))

    if optimal_ind is None:
        all_submission_dirs = [
            test_dir_fn(i) for i in range(num_simulations)
        ]

        mmpgps = MultiMPGadgetPowerSpec(
            all_submission_dirs,
            Latin_json,
            selected_ind=None,
            scale_factors=scale_factors,
        )

    else:
        assert num_simulations <= len(optimal_ind)

        all_submission_dirs = [
            test_dir_fn(i) for i in optimal_ind[:num_simulations]
        ]

        mmpgps = MultiMPGadgetPowerSpec(
            all_submission_dirs,
            Latin_json,
            optimal_ind[:num_simulations],
            scale_factors=scale_factors,
        )

    # Move into a folder
    os.makedirs("data", exist_ok=True)
    new_folder = os.path.join("data", h5_folder_name(num=num_simulations, npart=npart, box=box))
    os.makedirs(
        new_folder,
        exist_ok=True,
    )

    # Generate h5 file
    # Save into a folder
    mmpgps.create_hdf5(
        os.path.join(new_folder, hdf5_name)
    )

    with open(Latin_json, "r") as fload:
        param_dict = json.load(fload)

    with open(os.path.join(new_folder, "emulator_params.json"), "w") as fwrite:

        json.dump(param_dict, fwrite)

