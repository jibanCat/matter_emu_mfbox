"""
Prepare budget analysis data
"""
import os
from typing import Optional
import numpy as np

from matplotlib import pyplot as plt

from matter_multi_fidelity_emu.plottings.validation_loader import ValidationLoader
from matter_multi_fidelity_emu.data_loader import folder_name
from .make_validations_dgmgp import folder_name as dgmgp_folder

from .make_plots_pipeline import ar1_folder_name, nargp_folder_name, dgmgp_folder_name


class BudgetVloaders:
    """
    Prepare validation loaders
    """

    def __init__(self, img_dir: str = "data/output/"):
        old_dir = os.getcwd()
        os.chdir(img_dir)



        ############################ Loading all AR1+NARGP+dGMGP ############################
        # shared parameters
        res_l = 128
        res_h = 512
        box_l = 256
        box_l_2 = 100
        box_h = 256

        ## All redshifts and All LR + HR
        z       = [0, 0.2, 0.5, 1.0, 2.0, 3.0]
        all_lfs = [12, 18, 24, 30, 36, 42, 48, 54, 60, ]
        all_hfs = list(range(3, 19))

        available_slices = [57, 58, 59, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        all_slices       = [
            sorted(available_slices[:i]) for i in all_hfs
        ]

        for slice in all_slices:
            for num_lf in all_lfs:
                # dGMGP: Vary redshifts
                num_hf = len(slice)

                setattr(self, "dgmgp_L{}_L2box{}_H{}_z0_1_2_slice_{}".format(
                            num_lf, box_l_2, num_hf, "_".join(map(str, slice)
                        )
                    ), ValidationLoader(
                        [
                            dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
                        ],
                        num_lowres_list=[num_lf for _ in z],
                        num_highres=num_hf,
                        dGMGP=True,
                    )
                )
                setattr(getattr(self, "dgmgp_L{}_L2box{}_H{}_z0_1_2_slice_{}".format(
                            num_lf, box_l_2, num_hf, "_".join(map(str, slice)
                        )
                    )
                ), "res_l", res_l)
                setattr(getattr(self, "dgmgp_L{}_L2box{}_H{}_z0_1_2_slice_{}".format(
                            num_lf, box_l_2, num_hf, "_".join(map(str, slice)
                        )
                    )
                ), "res_h", res_h)
                setattr(getattr(self, "dgmgp_L{}_L2box{}_H{}_z0_1_2_slice_{}".format(
                            num_lf, box_l_2, num_hf, "_".join(map(str, slice)
                        )
                    )
                ), "box_l", box_l)
                setattr(getattr(self, "dgmgp_L{}_L2box{}_H{}_z0_1_2_slice_{}".format(
                            num_lf, box_l_2, num_hf, "_".join(map(str, slice)
                        )
                    )
                ), "box_h", box_h)
                setattr(getattr(self, "dgmgp_L{}_L2box{}_H{}_z0_1_2_slice_{}".format(
                            num_lf, box_l_2, num_hf, "_".join(map(str, slice)
                        )
                    )
                ), "z"    , z)
                setattr(getattr(self, "dgmgp_L{}_L2box{}_H{}_z0_1_2_slice_{}".format(
                            num_lf, box_l_2, num_hf, "_".join(map(str, slice)
                        )
                    )
                ), "slice", slice)
                setattr(getattr(self, "dgmgp_L{}_L2box{}_H{}_z0_1_2_slice_{}".format(
                            num_lf, box_l_2, num_hf, "_".join(map(str, slice)
                        )
                    )
                ), "num_lf", num_lf)
                setattr(getattr(self, "dgmgp_L{}_L2box{}_H{}_z0_1_2_slice_{}".format(
                            num_lf, box_l_2, num_hf, "_".join(map(str, slice)
                        )
                    )
                ), "num_hf", num_hf)
                
                # AR1: Vary redshifts
                setattr(self, "ar1_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice))
                ), ValidationLoader(
                        [
                            ar1_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
                        ],
                        num_lowres_list=[num_lf for _ in z],
                        num_highres=num_hf,
                    )
                )
                setattr(getattr(self, "ar1_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "res_l", res_l)
                setattr(getattr(self, "ar1_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "res_h", res_h)
                setattr(getattr(self, "ar1_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "box_l", box_l)
                setattr(getattr(self, "ar1_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "box_h", box_h)
                setattr(getattr(self, "ar1_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "z",     z )
                setattr(getattr(self, "ar1_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "slice", slice)
                setattr(getattr(self, "ar1_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "num_lf", num_lf)
                setattr(getattr(self, "ar1_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "num_hf", num_hf)

                # NARGP: Vary redshifts
                setattr(self, "nargp_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice))
                ), ValidationLoader(
                        [
                            nargp_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
                        ],
                        num_lowres_list=[num_lf for _ in z],
                        num_highres=num_hf,
                    )
                )
                setattr(getattr(self, "nargp_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "res_l", res_l)
                setattr(getattr(self, "nargp_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "res_h", res_h)
                setattr(getattr(self, "nargp_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "box_l", box_l)
                setattr(getattr(self, "nargp_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "box_h", box_h)
                setattr(getattr(self, "nargp_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "z"    , z)
                setattr(getattr(self, "nargp_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "slice", slice)
                setattr(getattr(self, "nargp_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "num_lf", num_lf)
                setattr(getattr(self, "nargp_L{}_H{}_z0_1_2_slice_{}".format(
                    num_lf, num_hf, "_".join(map(str, slice)))
                ), "num_hf", num_hf)

        # change back to original dir
        os.chdir(old_dir)
