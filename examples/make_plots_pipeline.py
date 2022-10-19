import os
from typing import Optional
import numpy as np

from matplotlib import pyplot as plt

from matter_multi_fidelity_emu.plottings.validation_loader import ValidationLoader
from matter_multi_fidelity_emu.data_loader import folder_name
from .make_validations_dgmgp import folder_name as dgmgp_folder

# folder_name
# e.g., Matterpower_24_res128box256_3_res512box256_z3_ind-57-58-59

# put the folder name on top, easier to find and modify
def ar1_folder_name(
    num1: int,
    res1: int,
    box1: int,
    num2: int,
    res2: int,
    box2: int,
    z: float,
    selected_ind,
):
    """
    As simple as, folder_name + "AR1/"
    """
    return os.path.join(folder_name(
        num1,
        res1,
        box1,
        num2,
        res2,
        box2,
        z,
        selected_ind,
    ), "AR1")

def nargp_folder_name(
    num1: int,
    res1: int,
    box1: int,
    num2: int,
    res2: int,
    box2: int,
    z: float,
    selected_ind,
):
    """
    As simple as, folder_name + "AR1/"
    """
    return os.path.join(folder_name(
        num1,
        res1,
        box1,
        num2,
        res2,
        box2,
        z,
        selected_ind,
    ), "NARGP")


def dgmgp_folder_name(
    num1: int,
    res1: int,
    box1: int,

    num2: int,
    res2: int,
    box2: int,

    num3: int,
    res3: int,
    box3: int,
    z: float,
    selected_ind,
):
    return os.path.join(dgmgp_folder(
        num1,
        res1,
        box1,

        num2,
        res2,
        box2,

        num3,
        res3,
        box3,
        z,
        selected_ind
        ),
        "dGMGP",
)




class PreloadedVloaders:
    """
    Prepare validation loaders
    """

    def __init__(self, img_dir: str = "data/output/"):
        old_dir = os.getcwd()
        os.chdir(img_dir)

        ####################### Vary LF #######################
        # AR1: z = 0
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 3
        self.ar1_H3_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H3_slice19.res_l = res_l
        self.ar1_H3_slice19.res_h = res_h
        self.ar1_H3_slice19.box_l = box_l
        self.ar1_H3_slice19.box_h = box_h
        self.ar1_H3_slice19.z     = z 
        self.ar1_H3_slice19.slice = slice
        self.ar1_H3_slice19.num_lf = num_lf
        self.ar1_H3_slice19.num_hf = num_hf
        # z = 0.2
        z = 0.2
        self.ar1_H3_slice19_z0_2 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H3_slice19_z0_2.res_l = res_l
        self.ar1_H3_slice19_z0_2.res_h = res_h
        self.ar1_H3_slice19_z0_2.box_l = box_l
        self.ar1_H3_slice19_z0_2.box_h = box_h
        self.ar1_H3_slice19_z0_2.z     = z 
        self.ar1_H3_slice19_z0_2.slice = slice
        self.ar1_H3_slice19_z0_2.num_lf = num_lf
        self.ar1_H3_slice19_z0_2.num_hf = num_hf
        # z = 0.5
        z = 0.5
        self.ar1_H3_slice19_z0_5 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H3_slice19_z0_5.res_l = res_l
        self.ar1_H3_slice19_z0_5.res_h = res_h
        self.ar1_H3_slice19_z0_5.box_l = box_l
        self.ar1_H3_slice19_z0_5.box_h = box_h
        self.ar1_H3_slice19_z0_5.z     = z 
        self.ar1_H3_slice19_z0_5.slice = slice
        self.ar1_H3_slice19_z0_5.num_lf = num_lf
        self.ar1_H3_slice19_z0_5.num_hf = num_hf
        # z = 1
        z = 1
        self.ar1_H3_slice19_z1 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H3_slice19_z1.res_l = res_l
        self.ar1_H3_slice19_z1.res_h = res_h
        self.ar1_H3_slice19_z1.box_l = box_l
        self.ar1_H3_slice19_z1.box_h = box_h
        self.ar1_H3_slice19_z1.z     = z 
        self.ar1_H3_slice19_z1.slice = slice
        self.ar1_H3_slice19_z1.num_lf = num_lf
        self.ar1_H3_slice19_z1.num_hf = num_hf
        # z = 2
        z = 2
        self.ar1_H3_slice19_z2 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H3_slice19_z2.res_l = res_l
        self.ar1_H3_slice19_z2.res_h = res_h
        self.ar1_H3_slice19_z2.box_l = box_l
        self.ar1_H3_slice19_z2.box_h = box_h
        self.ar1_H3_slice19_z2.z     = z 
        self.ar1_H3_slice19_z2.slice = slice
        self.ar1_H3_slice19_z2.num_lf = num_lf
        self.ar1_H3_slice19_z2.num_hf = num_hf
        # z = 3
        z = 3
        self.ar1_H3_slice19_z3 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H3_slice19_z3.res_l = res_l
        self.ar1_H3_slice19_z3.res_h = res_h
        self.ar1_H3_slice19_z3.box_l = box_l
        self.ar1_H3_slice19_z3.box_h = box_h
        self.ar1_H3_slice19_z3.z     = z 
        self.ar1_H3_slice19_z3.slice = slice
        self.ar1_H3_slice19_z3.num_lf = num_lf
        self.ar1_H3_slice19_z3.num_hf = num_hf

        # dGMGP: vary LF
        # z = 0
        res_l = 128
        res_h = 512
        box_l_1 = 256
        box_l_2 = 100
        box_h = 256
        z = 0
        slice = [57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 3
        self.dgmgp_H3_slice19 = ValidationLoader(
            [
                dgmgp_folder_name(n_lf, res_l, box_l_1, n_lf, res_l, box_l_2, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_H3_slice19.res_l = res_l
        self.dgmgp_H3_slice19.res_h = res_h
        self.dgmgp_H3_slice19.box_l = box_l
        self.dgmgp_H3_slice19.box_l_2 = box_l_2
        self.dgmgp_H3_slice19.box_h = box_h
        self.dgmgp_H3_slice19.z     = z 
        self.dgmgp_H3_slice19.slice = slice
        self.dgmgp_H3_slice19.num_lf = num_lf
        self.dgmgp_H3_slice19.num_hf = num_hf
        # z = 0.2
        z = 0.2
        self.dgmgp_H3_slice19_z0_2 = ValidationLoader(
            [
                dgmgp_folder_name(n_lf, res_l, box_l_1, n_lf, res_l, box_l_2, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_H3_slice19_z0_2.res_l = res_l
        self.dgmgp_H3_slice19_z0_2.res_h = res_h
        self.dgmgp_H3_slice19_z0_2.box_l = box_l
        self.dgmgp_H3_slice19_z0_2.box_l_2 = box_l_2
        self.dgmgp_H3_slice19_z0_2.box_h = box_h
        self.dgmgp_H3_slice19_z0_2.z     = z 
        self.dgmgp_H3_slice19_z0_2.slice = slice
        self.dgmgp_H3_slice19_z0_2.num_lf = num_lf
        self.dgmgp_H3_slice19_z0_2.num_hf = num_hf
        # z = 0.5
        z = 0.5
        self.dgmgp_H3_slice19_z0_5 = ValidationLoader(
            [
                dgmgp_folder_name(n_lf, res_l, box_l_1, n_lf, res_l, box_l_2, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_H3_slice19_z0_5.res_l = res_l
        self.dgmgp_H3_slice19_z0_5.res_h = res_h
        self.dgmgp_H3_slice19_z0_5.box_l = box_l
        self.dgmgp_H3_slice19_z0_5.box_l_2 = box_l_2
        self.dgmgp_H3_slice19_z0_5.box_h = box_h
        self.dgmgp_H3_slice19_z0_5.z     = z 
        self.dgmgp_H3_slice19_z0_5.slice = slice
        self.dgmgp_H3_slice19_z0_5.num_lf = num_lf
        self.dgmgp_H3_slice19_z0_5.num_hf = num_hf
        # z = 1
        z = 1
        self.dgmgp_H3_slice19_z1 = ValidationLoader(
            [
                dgmgp_folder_name(n_lf, res_l, box_l_1, n_lf, res_l, box_l_2, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_H3_slice19_z1.res_l = res_l
        self.dgmgp_H3_slice19_z1.res_h = res_h
        self.dgmgp_H3_slice19_z1.box_l = box_l
        self.dgmgp_H3_slice19_z1.box_l_2 = box_l_2
        self.dgmgp_H3_slice19_z1.box_h = box_h
        self.dgmgp_H3_slice19_z1.z     = z 
        self.dgmgp_H3_slice19_z1.slice = slice
        self.dgmgp_H3_slice19_z1.num_lf = num_lf
        self.dgmgp_H3_slice19_z1.num_hf = num_hf
        # z = 2
        z = 2
        self.dgmgp_H3_slice19_z2 = ValidationLoader(
            [
                dgmgp_folder_name(n_lf, res_l, box_l_1, n_lf, res_l, box_l_2, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_H3_slice19_z2.res_l = res_l
        self.dgmgp_H3_slice19_z2.res_h = res_h
        self.dgmgp_H3_slice19_z2.box_l = box_l
        self.dgmgp_H3_slice19_z2.box_l_2 = box_l_2
        self.dgmgp_H3_slice19_z2.box_h = box_h
        self.dgmgp_H3_slice19_z2.z     = z 
        self.dgmgp_H3_slice19_z2.slice = slice
        self.dgmgp_H3_slice19_z2.num_lf = num_lf
        self.dgmgp_H3_slice19_z2.num_hf = num_hf
        # z = 3
        z = 3
        self.dgmgp_H3_slice19_z3 = ValidationLoader(
            [
                dgmgp_folder_name(n_lf, res_l, box_l_1, n_lf, res_l, box_l_2, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_H3_slice19_z3.res_l = res_l
        self.dgmgp_H3_slice19_z3.res_h = res_h
        self.dgmgp_H3_slice19_z3.box_l = box_l
        self.dgmgp_H3_slice19_z3.box_l_2 = box_l_2
        self.dgmgp_H3_slice19_z3.box_h = box_h
        self.dgmgp_H3_slice19_z3.z     = z 
        self.dgmgp_H3_slice19_z3.slice = slice
        self.dgmgp_H3_slice19_z3.num_lf = num_lf
        self.dgmgp_H3_slice19_z3.num_hf = num_hf

        # NARGP: vary LF
        # z = 0
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60] # assume L1 and L2 have the same L points
        num_hf = 3
        self.nargp_H3_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H3_slice19.res_l = res_l
        self.nargp_H3_slice19.res_h = res_h
        self.nargp_H3_slice19.box_l = box_l
        self.nargp_H3_slice19.box_h = box_h
        self.nargp_H3_slice19.z     = z 
        self.nargp_H3_slice19.slice = slice
        self.nargp_H3_slice19.num_lf = num_lf
        self.nargp_H3_slice19.num_hf = num_hf
        # z = 0.2
        z = 0.2
        self.nargp_H3_slice19_z0_2 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H3_slice19_z0_2.res_l = res_l
        self.nargp_H3_slice19_z0_2.res_h = res_h
        self.nargp_H3_slice19_z0_2.box_l = box_l
        self.nargp_H3_slice19_z0_2.box_h = box_h
        self.nargp_H3_slice19_z0_2.z     = z 
        self.nargp_H3_slice19_z0_2.slice = slice
        self.nargp_H3_slice19_z0_2.num_lf = num_lf
        self.nargp_H3_slice19_z0_2.num_hf = num_hf
        # z = 0.5
        z = 0.5
        self.nargp_H3_slice19_z0_5 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H3_slice19_z0_5.res_l = res_l
        self.nargp_H3_slice19_z0_5.res_h = res_h
        self.nargp_H3_slice19_z0_5.box_l = box_l
        self.nargp_H3_slice19_z0_5.box_h = box_h
        self.nargp_H3_slice19_z0_5.z     = z 
        self.nargp_H3_slice19_z0_5.slice = slice
        self.nargp_H3_slice19_z0_5.num_lf = num_lf
        self.nargp_H3_slice19_z0_5.num_hf = num_hf
        # z = 1
        z = 1
        self.nargp_H3_slice19_z1 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H3_slice19_z1.res_l = res_l
        self.nargp_H3_slice19_z1.res_h = res_h
        self.nargp_H3_slice19_z1.box_l = box_l
        self.nargp_H3_slice19_z1.box_h = box_h
        self.nargp_H3_slice19_z1.z     = z 
        self.nargp_H3_slice19_z1.slice = slice
        self.nargp_H3_slice19_z1.num_lf = num_lf
        self.nargp_H3_slice19_z1.num_hf = num_hf
        # z = 2
        z = 2
        self.nargp_H3_slice19_z2 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H3_slice19_z2.res_l = res_l
        self.nargp_H3_slice19_z2.res_h = res_h
        self.nargp_H3_slice19_z2.box_l = box_l
        self.nargp_H3_slice19_z2.box_h = box_h
        self.nargp_H3_slice19_z2.z     = z 
        self.nargp_H3_slice19_z2.slice = slice
        self.nargp_H3_slice19_z2.num_lf = num_lf
        self.nargp_H3_slice19_z2.num_hf = num_hf
        # z = 3
        z = 3
        self.nargp_H3_slice19_z3 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H3_slice19_z3.res_l = res_l
        self.nargp_H3_slice19_z3.res_h = res_h
        self.nargp_H3_slice19_z3.box_l = box_l
        self.nargp_H3_slice19_z3.box_h = box_h
        self.nargp_H3_slice19_z3.z     = z 
        self.nargp_H3_slice19_z3.slice = slice
        self.nargp_H3_slice19_z3.num_lf = num_lf
        self.nargp_H3_slice19_z3.num_hf = num_hf


        # AR1: Vary redshifts
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = [0, 0.2, 0.5, 1.0, 2.0, 3.0] # Forgot to run z=0.5
        # z = [0, 0.2, 1.0, 2.0, 3.0]
        slice = [57, 58, 59]
        num_lf = 60
        num_hf = 3
        self.ar1_L60_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                ar1_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.ar1_L60_H3_z0_1_2_slice_19.res_l = res_l
        self.ar1_L60_H3_z0_1_2_slice_19.res_h = res_h
        self.ar1_L60_H3_z0_1_2_slice_19.box_l = box_l
        self.ar1_L60_H3_z0_1_2_slice_19.box_h = box_h
        self.ar1_L60_H3_z0_1_2_slice_19.z     = z 
        self.ar1_L60_H3_z0_1_2_slice_19.slice = slice
        self.ar1_L60_H3_z0_1_2_slice_19.num_lf = num_lf
        self.ar1_L60_H3_z0_1_2_slice_19.num_hf = num_hf

        # NARGP: Vary redshifts
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = [0, 0.2, 0.5, 1.0, 2.0, 3.0] # Forgot to run z=0.5
        # z = [0, 0.2, 1.0, 2.0, 3.0]
        slice = [57, 58, 59]
        num_lf = 60
        num_hf = 3
        self.nargp_L60_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                nargp_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.nargp_L60_H3_z0_1_2_slice_19.res_l = res_l
        self.nargp_L60_H3_z0_1_2_slice_19.res_h = res_h
        self.nargp_L60_H3_z0_1_2_slice_19.box_l = box_l
        self.nargp_L60_H3_z0_1_2_slice_19.box_h = box_h
        self.nargp_L60_H3_z0_1_2_slice_19.z     = z 
        self.nargp_L60_H3_z0_1_2_slice_19.slice = slice
        self.nargp_L60_H3_z0_1_2_slice_19.num_lf = num_lf
        self.nargp_L60_H3_z0_1_2_slice_19.num_hf = num_hf

        # dGMGP: Vary redshifts
        res_l = 128
        res_h = 512
        box_l = 256
        box_l_2 = 100
        box_h = 256
        z = [0, 0.2, 0.5, 1.0, 2.0, 3.0] # Forgot to run z=0.5
        # z = [0, 0.2, 1.0, 2.0, 3.0]
        slice = [57, 58, 59]
        num_lf = 60
        num_hf = 3
        self.dgmgp_L60_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_H3_z0_1_2_slice_19.res_l = res_l
        self.dgmgp_L60_H3_z0_1_2_slice_19.res_h = res_h
        self.dgmgp_L60_H3_z0_1_2_slice_19.box_l = box_l
        self.dgmgp_L60_H3_z0_1_2_slice_19.box_h = box_h
        self.dgmgp_L60_H3_z0_1_2_slice_19.z     = z 
        self.dgmgp_L60_H3_z0_1_2_slice_19.slice = slice
        self.dgmgp_L60_H3_z0_1_2_slice_19.num_lf = num_lf
        self.dgmgp_L60_H3_z0_1_2_slice_19.num_hf = num_hf

        ####################### Vary HF #######################
        ## Start Here:
        ## ----
        # AR1, z = 0: Vary HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [[57, 58, 59], [0, 57, 58, 59], [0, 1, 57, 58, 59], [0, 1, 2, 57, 58, 59], [0, 1, 2, 3, 57, 58, 59],
            [0, 1, 2, 3, 4, 57, 58, 59], [0, 1, 2, 3, 4, 5, 57, 58, 59], [0, 1, 2, 3, 4, 5, 6, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 57, 58, 59], [0, 1, 2, 3, 4, 5, 6, 7, 8, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 57, 58, 59], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 57, 58, 59],
        ]
        num_lf = [60 for _ in range(len(slice))]
        num_hf = [len(s) for s in slice]
        self.ar1_L60_H3_10_z0 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_L60_H3_10_z0.res_l = res_l
        self.ar1_L60_H3_10_z0.res_h = res_h
        self.ar1_L60_H3_10_z0.box_l = box_l
        self.ar1_L60_H3_10_z0.box_h = box_h
        self.ar1_L60_H3_10_z0.z     = z 
        self.ar1_L60_H3_10_z0.slice = slice
        self.ar1_L60_H3_10_z0.num_lf = num_lf
        self.ar1_L60_H3_10_z0.num_hf = num_hf
        # z = 0.2
        z = 0.2
        self.ar1_L60_H3_10_z0_2 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_L60_H3_10_z0_2.res_l = res_l
        self.ar1_L60_H3_10_z0_2.res_h = res_h
        self.ar1_L60_H3_10_z0_2.box_l = box_l
        self.ar1_L60_H3_10_z0_2.box_h = box_h
        self.ar1_L60_H3_10_z0_2.z     = z 
        self.ar1_L60_H3_10_z0_2.slice = slice
        self.ar1_L60_H3_10_z0_2.num_lf = num_lf
        self.ar1_L60_H3_10_z0_2.num_hf = num_hf
        # z = 0.5
        z = 0.5
        self.ar1_L60_H3_10_z0_5 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_L60_H3_10_z0_5.res_l = res_l
        self.ar1_L60_H3_10_z0_5.res_h = res_h
        self.ar1_L60_H3_10_z0_5.box_l = box_l
        self.ar1_L60_H3_10_z0_5.box_h = box_h
        self.ar1_L60_H3_10_z0_5.z     = z 
        self.ar1_L60_H3_10_z0_5.slice = slice
        self.ar1_L60_H3_10_z0_5.num_lf = num_lf
        self.ar1_L60_H3_10_z0_5.num_hf = num_hf
        # z = 1
        z = 1
        self.ar1_L60_H3_10_z1 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_L60_H3_10_z1.res_l = res_l
        self.ar1_L60_H3_10_z1.res_h = res_h
        self.ar1_L60_H3_10_z1.box_l = box_l
        self.ar1_L60_H3_10_z1.box_h = box_h
        self.ar1_L60_H3_10_z1.z     = z 
        self.ar1_L60_H3_10_z1.slice = slice
        self.ar1_L60_H3_10_z1.num_lf = num_lf
        self.ar1_L60_H3_10_z1.num_hf = num_hf
        # z = 2
        z = 2
        self.ar1_L60_H3_10_z2 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_L60_H3_10_z2.res_l = res_l
        self.ar1_L60_H3_10_z2.res_h = res_h
        self.ar1_L60_H3_10_z2.box_l = box_l
        self.ar1_L60_H3_10_z2.box_h = box_h
        self.ar1_L60_H3_10_z2.z     = z 
        self.ar1_L60_H3_10_z2.slice = slice
        self.ar1_L60_H3_10_z2.num_lf = num_lf
        self.ar1_L60_H3_10_z2.num_hf = num_hf
        # z = 3
        z = 3
        self.ar1_L60_H3_10_z3 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_L60_H3_10_z3.res_l = res_l
        self.ar1_L60_H3_10_z3.res_h = res_h
        self.ar1_L60_H3_10_z3.box_l = box_l
        self.ar1_L60_H3_10_z3.box_h = box_h
        self.ar1_L60_H3_10_z3.z     = z 
        self.ar1_L60_H3_10_z3.slice = slice
        self.ar1_L60_H3_10_z3.num_lf = num_lf
        self.ar1_L60_H3_10_z3.num_hf = num_hf

        # NARGP, z = 0: Vary HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [[57, 58, 59], [0, 57, 58, 59], [0, 1, 57, 58, 59], [0, 1, 2, 57, 58, 59], [0, 1, 2, 3, 57, 58, 59],
            [0, 1, 2, 3, 4, 57, 58, 59], [0, 1, 2, 3, 4, 5, 57, 58, 59], [0, 1, 2, 3, 4, 5, 6, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 57, 58, 59], [0, 1, 2, 3, 4, 5, 6, 7, 8, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 57, 58, 59], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 57, 58, 59],
        ]
        num_lf = [60 for _ in range(len(slice))]
        num_hf = [len(s) for s in slice]
        self.nargp_L60_H3_10_z0 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_L60_H3_10_z0.res_l = res_l
        self.nargp_L60_H3_10_z0.res_h = res_h
        self.nargp_L60_H3_10_z0.box_l = box_l
        self.nargp_L60_H3_10_z0.box_h = box_h
        self.nargp_L60_H3_10_z0.z     = z 
        self.nargp_L60_H3_10_z0.slice = slice
        self.nargp_L60_H3_10_z0.num_lf = num_lf
        self.nargp_L60_H3_10_z0.num_hf = num_hf
        # z = 0.2
        z = 0.2
        self.nargp_L60_H3_10_z0_2 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_L60_H3_10_z0_2.res_l = res_l
        self.nargp_L60_H3_10_z0_2.res_h = res_h
        self.nargp_L60_H3_10_z0_2.box_l = box_l
        self.nargp_L60_H3_10_z0_2.box_h = box_h
        self.nargp_L60_H3_10_z0_2.z     = z 
        self.nargp_L60_H3_10_z0_2.slice = slice
        self.nargp_L60_H3_10_z0_2.num_lf = num_lf
        self.nargp_L60_H3_10_z0_2.num_hf = num_hf
        # z = 0.5
        z = 0.5
        self.nargp_L60_H3_10_z0_5 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_L60_H3_10_z0_5.res_l = res_l
        self.nargp_L60_H3_10_z0_5.res_h = res_h
        self.nargp_L60_H3_10_z0_5.box_l = box_l
        self.nargp_L60_H3_10_z0_5.box_h = box_h
        self.nargp_L60_H3_10_z0_5.z     = z 
        self.nargp_L60_H3_10_z0_5.slice = slice
        self.nargp_L60_H3_10_z0_5.num_lf = num_lf
        self.nargp_L60_H3_10_z0_5.num_hf = num_hf
        # z = 1
        z = 1
        self.nargp_L60_H3_10_z1 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_L60_H3_10_z1.res_l = res_l
        self.nargp_L60_H3_10_z1.res_h = res_h
        self.nargp_L60_H3_10_z1.box_l = box_l
        self.nargp_L60_H3_10_z1.box_h = box_h
        self.nargp_L60_H3_10_z1.z     = z 
        self.nargp_L60_H3_10_z1.slice = slice
        self.nargp_L60_H3_10_z1.num_lf = num_lf
        self.nargp_L60_H3_10_z1.num_hf = num_hf
        # z = 2
        z = 2
        self.nargp_L60_H3_10_z2 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_L60_H3_10_z2.res_l = res_l
        self.nargp_L60_H3_10_z2.res_h = res_h
        self.nargp_L60_H3_10_z2.box_l = box_l
        self.nargp_L60_H3_10_z2.box_h = box_h
        self.nargp_L60_H3_10_z2.z     = z 
        self.nargp_L60_H3_10_z2.slice = slice
        self.nargp_L60_H3_10_z2.num_lf = num_lf
        self.nargp_L60_H3_10_z2.num_hf = num_hf
        # z = 3
        z = 3
        self.nargp_L60_H3_10_z3 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_L60_H3_10_z3.res_l = res_l
        self.nargp_L60_H3_10_z3.res_h = res_h
        self.nargp_L60_H3_10_z3.box_l = box_l
        self.nargp_L60_H3_10_z3.box_h = box_h
        self.nargp_L60_H3_10_z3.z     = z 
        self.nargp_L60_H3_10_z3.slice = slice
        self.nargp_L60_H3_10_z3.num_lf = num_lf
        self.nargp_L60_H3_10_z3.num_hf = num_hf

        # dGMGP, z = 0: Vary HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_l_2 = 100
        box_h = 256
        z = 0
        slice = [[57, 58, 59], [0, 57, 58, 59], [0, 1, 57, 58, 59], [0, 1, 2, 57, 58, 59], [0, 1, 2, 3, 57, 58, 59],
            [0, 1, 2, 3, 4, 57, 58, 59], [0, 1, 2, 3, 4, 5, 57, 58, 59], [0, 1, 2, 3, 4, 5, 6, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 57, 58, 59], [0, 1, 2, 3, 4, 5, 6, 7, 8, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 57, 58, 59], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 57, 58, 59],
        ]
        num_lf = [60 for _ in range(len(slice))]
        num_hf = [len(s) for s in slice]
        self.dgmgp_L60_H3_10_z0 = ValidationLoader(
            [
                dgmgp_folder_name(n_lf, res_l, box_l, n_lf, res_l, box_l_2, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_H3_10_z0.res_l = res_l
        self.dgmgp_L60_H3_10_z0.res_h = res_h
        self.dgmgp_L60_H3_10_z0.box_l = box_l
        self.dgmgp_L60_H3_10_z0.box_l_2 = box_l_2
        self.dgmgp_L60_H3_10_z0.box_h = box_h
        self.dgmgp_L60_H3_10_z0.z     = z 
        self.dgmgp_L60_H3_10_z0.slice = slice
        self.dgmgp_L60_H3_10_z0.num_lf = num_lf
        self.dgmgp_L60_H3_10_z0.num_hf = num_hf
        # z = 0.2
        z = 0.2
        self.dgmgp_L60_H3_10_z0_2 = ValidationLoader(
            [
                dgmgp_folder_name(n_lf, res_l, box_l, n_lf, res_l, box_l_2, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_H3_10_z0_2.res_l = res_l
        self.dgmgp_L60_H3_10_z0_2.res_h = res_h
        self.dgmgp_L60_H3_10_z0_2.box_l = box_l
        self.dgmgp_L60_H3_10_z0_2.box_l_2 = box_l_2
        self.dgmgp_L60_H3_10_z0_2.box_h = box_h
        self.dgmgp_L60_H3_10_z0_2.z     = z 
        self.dgmgp_L60_H3_10_z0_2.slice = slice
        self.dgmgp_L60_H3_10_z0_2.num_lf = num_lf
        self.dgmgp_L60_H3_10_z0_2.num_hf = num_hf
        # z = 0.5
        z = 0.5
        self.dgmgp_L60_H3_10_z0_5 = ValidationLoader(
            [
                dgmgp_folder_name(n_lf, res_l, box_l, n_lf, res_l, box_l_2, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_H3_10_z0_5.res_l = res_l
        self.dgmgp_L60_H3_10_z0_5.res_h = res_h
        self.dgmgp_L60_H3_10_z0_5.box_l = box_l
        self.dgmgp_L60_H3_10_z0_5.box_l_2 = box_l_2
        self.dgmgp_L60_H3_10_z0_5.box_h = box_h
        self.dgmgp_L60_H3_10_z0_5.z     = z 
        self.dgmgp_L60_H3_10_z0_5.slice = slice
        self.dgmgp_L60_H3_10_z0_5.num_lf = num_lf
        self.dgmgp_L60_H3_10_z0_5.num_hf = num_hf
        # z = 1
        z = 1
        self.dgmgp_L60_H3_10_z1 = ValidationLoader(
            [
                dgmgp_folder_name(n_lf, res_l, box_l, n_lf, res_l, box_l_2, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_H3_10_z1.res_l = res_l
        self.dgmgp_L60_H3_10_z1.res_h = res_h
        self.dgmgp_L60_H3_10_z1.box_l = box_l
        self.dgmgp_L60_H3_10_z1.box_l_2 = box_l_2
        self.dgmgp_L60_H3_10_z1.box_h = box_h
        self.dgmgp_L60_H3_10_z1.z     = z 
        self.dgmgp_L60_H3_10_z1.slice = slice
        self.dgmgp_L60_H3_10_z1.num_lf = num_lf
        self.dgmgp_L60_H3_10_z1.num_hf = num_hf
        # z = 2
        z = 2
        self.dgmgp_L60_H3_10_z2 = ValidationLoader(
            [
                dgmgp_folder_name(n_lf, res_l, box_l, n_lf, res_l, box_l_2, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_H3_10_z2.res_l = res_l
        self.dgmgp_L60_H3_10_z2.res_h = res_h
        self.dgmgp_L60_H3_10_z2.box_l = box_l
        self.dgmgp_L60_H3_10_z2.box_l_2 = box_l_2
        self.dgmgp_L60_H3_10_z2.box_h = box_h
        self.dgmgp_L60_H3_10_z2.z     = z 
        self.dgmgp_L60_H3_10_z2.slice = slice
        self.dgmgp_L60_H3_10_z2.num_lf = num_lf
        self.dgmgp_L60_H3_10_z2.num_hf = num_hf
        # z = 3
        z = 3
        self.dgmgp_L60_H3_10_z3 = ValidationLoader(
            [
                dgmgp_folder_name(n_lf, res_l, box_l, n_lf, res_l, box_l_2, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_H3_10_z3.res_l = res_l
        self.dgmgp_L60_H3_10_z3.res_h = res_h
        self.dgmgp_L60_H3_10_z3.box_l = box_l
        self.dgmgp_L60_H3_10_z3.box_l_2 = box_l_2
        self.dgmgp_L60_H3_10_z3.box_h = box_h
        self.dgmgp_L60_H3_10_z3.z     = z 
        self.dgmgp_L60_H3_10_z3.slice = slice
        self.dgmgp_L60_H3_10_z3.num_lf = num_lf
        self.dgmgp_L60_H3_10_z3.num_hf = num_hf

        ## Vary AR1's HR points
        ## Start Here
        ## -----

        # AR1: 2 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [57, 58]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 2
        self.ar1_H2_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H2_slice19.res_l = res_l
        self.ar1_H2_slice19.res_h = res_h
        self.ar1_H2_slice19.box_l = box_l
        self.ar1_H2_slice19.box_h = box_h
        self.ar1_H2_slice19.z     = z 
        self.ar1_H2_slice19.slice = slice
        self.ar1_H2_slice19.num_lf = num_lf
        self.ar1_H2_slice19.num_hf = num_hf

        # AR1: 4 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 4
        self.ar1_H4_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H4_slice19.res_l = res_l
        self.ar1_H4_slice19.res_h = res_h
        self.ar1_H4_slice19.box_l = box_l
        self.ar1_H4_slice19.box_h = box_h
        self.ar1_H4_slice19.z     = z 
        self.ar1_H4_slice19.slice = slice
        self.ar1_H4_slice19.num_lf = num_lf
        self.ar1_H4_slice19.num_hf = num_hf

        # AR1: 5 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 5
        self.ar1_H5_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H5_slice19.res_l = res_l
        self.ar1_H5_slice19.res_h = res_h
        self.ar1_H5_slice19.box_l = box_l
        self.ar1_H5_slice19.box_h = box_h
        self.ar1_H5_slice19.z     = z 
        self.ar1_H5_slice19.slice = slice
        self.ar1_H5_slice19.num_lf = num_lf
        self.ar1_H5_slice19.num_hf = num_hf

        # AR1: 6 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 6
        self.ar1_H6_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H6_slice19.res_l = res_l
        self.ar1_H6_slice19.res_h = res_h
        self.ar1_H6_slice19.box_l = box_l
        self.ar1_H6_slice19.box_h = box_h
        self.ar1_H6_slice19.z     = z 
        self.ar1_H6_slice19.slice = slice
        self.ar1_H6_slice19.num_lf = num_lf
        self.ar1_H6_slice19.num_hf = num_hf

        # AR1: 7 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 7
        self.ar1_H7_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H7_slice19.res_l = res_l
        self.ar1_H7_slice19.res_h = res_h
        self.ar1_H7_slice19.box_l = box_l
        self.ar1_H7_slice19.box_h = box_h
        self.ar1_H7_slice19.z     = z 
        self.ar1_H7_slice19.slice = slice
        self.ar1_H7_slice19.num_lf = num_lf
        self.ar1_H7_slice19.num_hf = num_hf

        # AR1: 8 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 8
        self.ar1_H8_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H8_slice19.res_l = res_l
        self.ar1_H8_slice19.res_h = res_h
        self.ar1_H8_slice19.box_l = box_l
        self.ar1_H8_slice19.box_h = box_h
        self.ar1_H8_slice19.z     = z 
        self.ar1_H8_slice19.slice = slice
        self.ar1_H8_slice19.num_lf = num_lf
        self.ar1_H8_slice19.num_hf = num_hf


        # AR1: 9 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 9
        self.ar1_H9_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H9_slice19.res_l = res_l
        self.ar1_H9_slice19.res_h = res_h
        self.ar1_H9_slice19.box_l = box_l
        self.ar1_H9_slice19.box_h = box_h
        self.ar1_H9_slice19.z     = z 
        self.ar1_H9_slice19.slice = slice
        self.ar1_H9_slice19.num_lf = num_lf
        self.ar1_H9_slice19.num_hf = num_hf

        # AR1: 10 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 10
        self.ar1_H10_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H10_slice19.res_l = res_l
        self.ar1_H10_slice19.res_h = res_h
        self.ar1_H10_slice19.box_l = box_l
        self.ar1_H10_slice19.box_h = box_h
        self.ar1_H10_slice19.z     = z 
        self.ar1_H10_slice19.slice = slice
        self.ar1_H10_slice19.num_lf = num_lf
        self.ar1_H10_slice19.num_hf = num_hf

        # AR1: 11 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 11
        self.ar1_H11_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H11_slice19.res_l = res_l
        self.ar1_H11_slice19.res_h = res_h
        self.ar1_H11_slice19.box_l = box_l
        self.ar1_H11_slice19.box_h = box_h
        self.ar1_H11_slice19.z     = z 
        self.ar1_H11_slice19.slice = slice
        self.ar1_H11_slice19.num_lf = num_lf
        self.ar1_H11_slice19.num_hf = num_hf

        # AR1: 12 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 12
        self.ar1_H12_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H12_slice19.res_l = res_l
        self.ar1_H12_slice19.res_h = res_h
        self.ar1_H12_slice19.box_l = box_l
        self.ar1_H12_slice19.box_h = box_h
        self.ar1_H12_slice19.z     = z 
        self.ar1_H12_slice19.slice = slice
        self.ar1_H12_slice19.num_lf = num_lf
        self.ar1_H12_slice19.num_hf = num_hf

        # AR1: 13 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 13
        self.ar1_H13_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H13_slice19.res_l = res_l
        self.ar1_H13_slice19.res_h = res_h
        self.ar1_H13_slice19.box_l = box_l
        self.ar1_H13_slice19.box_h = box_h
        self.ar1_H13_slice19.z     = z 
        self.ar1_H13_slice19.slice = slice
        self.ar1_H13_slice19.num_lf = num_lf
        self.ar1_H13_slice19.num_hf = num_hf

        # AR1: 14 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 14
        self.ar1_H14_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H14_slice19.res_l = res_l
        self.ar1_H14_slice19.res_h = res_h
        self.ar1_H14_slice19.box_l = box_l
        self.ar1_H14_slice19.box_h = box_h
        self.ar1_H14_slice19.z     = z 
        self.ar1_H14_slice19.slice = slice
        self.ar1_H14_slice19.num_lf = num_lf
        self.ar1_H14_slice19.num_hf = num_hf


        # AR1: 15 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 15
        self.ar1_H15_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H15_slice19.res_l = res_l
        self.ar1_H15_slice19.res_h = res_h
        self.ar1_H15_slice19.box_l = box_l
        self.ar1_H15_slice19.box_h = box_h
        self.ar1_H15_slice19.z     = z 
        self.ar1_H15_slice19.slice = slice
        self.ar1_H15_slice19.num_lf = num_lf
        self.ar1_H15_slice19.num_hf = num_hf


        # AR1: 16 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 16
        self.ar1_H16_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H16_slice19.res_l = res_l
        self.ar1_H16_slice19.res_h = res_h
        self.ar1_H16_slice19.box_l = box_l
        self.ar1_H16_slice19.box_h = box_h
        self.ar1_H16_slice19.z     = z 
        self.ar1_H16_slice19.slice = slice
        self.ar1_H16_slice19.num_lf = num_lf
        self.ar1_H16_slice19.num_hf = num_hf

        # AR1: 17 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 17
        self.ar1_H17_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H17_slice19.res_l = res_l
        self.ar1_H17_slice19.res_h = res_h
        self.ar1_H17_slice19.box_l = box_l
        self.ar1_H17_slice19.box_h = box_h
        self.ar1_H17_slice19.z     = z 
        self.ar1_H17_slice19.slice = slice
        self.ar1_H17_slice19.num_lf = num_lf
        self.ar1_H17_slice19.num_hf = num_hf

        # AR1: 18 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 18
        self.ar1_H18_slice19 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H18_slice19.res_l = res_l
        self.ar1_H18_slice19.res_h = res_h
        self.ar1_H18_slice19.box_l = box_l
        self.ar1_H18_slice19.box_h = box_h
        self.ar1_H18_slice19.z     = z 
        self.ar1_H18_slice19.slice = slice
        self.ar1_H18_slice19.num_lf = num_lf
        self.ar1_H18_slice19.num_hf = num_hf

        ## Vary NARGP's HR points
        ## Start Here
        ## -----

        # NARGP: 2 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [57, 58]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 2
        self.nargp_H2_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H2_slice19.res_l = res_l
        self.nargp_H2_slice19.res_h = res_h
        self.nargp_H2_slice19.box_l = box_l
        self.nargp_H2_slice19.box_h = box_h
        self.nargp_H2_slice19.z     = z 
        self.nargp_H2_slice19.slice = slice
        self.nargp_H2_slice19.num_lf = num_lf
        self.nargp_H2_slice19.num_hf = num_hf

        # NARGP: 4 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 4
        self.nargp_H4_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H4_slice19.res_l = res_l
        self.nargp_H4_slice19.res_h = res_h
        self.nargp_H4_slice19.box_l = box_l
        self.nargp_H4_slice19.box_h = box_h
        self.nargp_H4_slice19.z     = z 
        self.nargp_H4_slice19.slice = slice
        self.nargp_H4_slice19.num_lf = num_lf
        self.nargp_H4_slice19.num_hf = num_hf

        # NARGP: 5 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 5
        self.nargp_H5_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H5_slice19.res_l = res_l
        self.nargp_H5_slice19.res_h = res_h
        self.nargp_H5_slice19.box_l = box_l
        self.nargp_H5_slice19.box_h = box_h
        self.nargp_H5_slice19.z     = z 
        self.nargp_H5_slice19.slice = slice
        self.nargp_H5_slice19.num_lf = num_lf
        self.nargp_H5_slice19.num_hf = num_hf

        # NARGP: 6 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 6
        self.nargp_H6_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H6_slice19.res_l = res_l
        self.nargp_H6_slice19.res_h = res_h
        self.nargp_H6_slice19.box_l = box_l
        self.nargp_H6_slice19.box_h = box_h
        self.nargp_H6_slice19.z     = z 
        self.nargp_H6_slice19.slice = slice
        self.nargp_H6_slice19.num_lf = num_lf
        self.nargp_H6_slice19.num_hf = num_hf

        # NARGP: 7 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 7
        self.nargp_H7_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H7_slice19.res_l = res_l
        self.nargp_H7_slice19.res_h = res_h
        self.nargp_H7_slice19.box_l = box_l
        self.nargp_H7_slice19.box_h = box_h
        self.nargp_H7_slice19.z     = z 
        self.nargp_H7_slice19.slice = slice
        self.nargp_H7_slice19.num_lf = num_lf
        self.nargp_H7_slice19.num_hf = num_hf

        # NARGP: 8 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 8
        self.nargp_H8_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H8_slice19.res_l = res_l
        self.nargp_H8_slice19.res_h = res_h
        self.nargp_H8_slice19.box_l = box_l
        self.nargp_H8_slice19.box_h = box_h
        self.nargp_H8_slice19.z     = z 
        self.nargp_H8_slice19.slice = slice
        self.nargp_H8_slice19.num_lf = num_lf
        self.nargp_H8_slice19.num_hf = num_hf


        # NARGP: 9 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 9
        self.nargp_H9_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H9_slice19.res_l = res_l
        self.nargp_H9_slice19.res_h = res_h
        self.nargp_H9_slice19.box_l = box_l
        self.nargp_H9_slice19.box_h = box_h
        self.nargp_H9_slice19.z     = z 
        self.nargp_H9_slice19.slice = slice
        self.nargp_H9_slice19.num_lf = num_lf
        self.nargp_H9_slice19.num_hf = num_hf

        # NARGP: 10 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 10
        self.nargp_H10_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H10_slice19.res_l = res_l
        self.nargp_H10_slice19.res_h = res_h
        self.nargp_H10_slice19.box_l = box_l
        self.nargp_H10_slice19.box_h = box_h
        self.nargp_H10_slice19.z     = z 
        self.nargp_H10_slice19.slice = slice
        self.nargp_H10_slice19.num_lf = num_lf
        self.nargp_H10_slice19.num_hf = num_hf

        # NARGP: 11 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 11
        self.nargp_H11_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H11_slice19.res_l = res_l
        self.nargp_H11_slice19.res_h = res_h
        self.nargp_H11_slice19.box_l = box_l
        self.nargp_H11_slice19.box_h = box_h
        self.nargp_H11_slice19.z     = z 
        self.nargp_H11_slice19.slice = slice
        self.nargp_H11_slice19.num_lf = num_lf
        self.nargp_H11_slice19.num_hf = num_hf

        # NARGP: 12 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 12
        self.nargp_H12_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H12_slice19.res_l = res_l
        self.nargp_H12_slice19.res_h = res_h
        self.nargp_H12_slice19.box_l = box_l
        self.nargp_H12_slice19.box_h = box_h
        self.nargp_H12_slice19.z     = z 
        self.nargp_H12_slice19.slice = slice
        self.nargp_H12_slice19.num_lf = num_lf
        self.nargp_H12_slice19.num_hf = num_hf

        # NARGP: 13 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 13
        self.nargp_H13_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H13_slice19.res_l = res_l
        self.nargp_H13_slice19.res_h = res_h
        self.nargp_H13_slice19.box_l = box_l
        self.nargp_H13_slice19.box_h = box_h
        self.nargp_H13_slice19.z     = z 
        self.nargp_H13_slice19.slice = slice
        self.nargp_H13_slice19.num_lf = num_lf
        self.nargp_H13_slice19.num_hf = num_hf

        # NARGP: 14 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 14
        self.nargp_H14_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H14_slice19.res_l = res_l
        self.nargp_H14_slice19.res_h = res_h
        self.nargp_H14_slice19.box_l = box_l
        self.nargp_H14_slice19.box_h = box_h
        self.nargp_H14_slice19.z     = z 
        self.nargp_H14_slice19.slice = slice
        self.nargp_H14_slice19.num_lf = num_lf
        self.nargp_H14_slice19.num_hf = num_hf


        # NARGP: 15 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 15
        self.nargp_H15_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H15_slice19.res_l = res_l
        self.nargp_H15_slice19.res_h = res_h
        self.nargp_H15_slice19.box_l = box_l
        self.nargp_H15_slice19.box_h = box_h
        self.nargp_H15_slice19.z     = z 
        self.nargp_H15_slice19.slice = slice
        self.nargp_H15_slice19.num_lf = num_lf
        self.nargp_H15_slice19.num_hf = num_hf


        # NARGP: 16 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 16
        self.nargp_H16_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H16_slice19.res_l = res_l
        self.nargp_H16_slice19.res_h = res_h
        self.nargp_H16_slice19.box_l = box_l
        self.nargp_H16_slice19.box_h = box_h
        self.nargp_H16_slice19.z     = z 
        self.nargp_H16_slice19.slice = slice
        self.nargp_H16_slice19.num_lf = num_lf
        self.nargp_H16_slice19.num_hf = num_hf

        # NARGP: 17 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 17
        self.nargp_H17_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H17_slice19.res_l = res_l
        self.nargp_H17_slice19.res_h = res_h
        self.nargp_H17_slice19.box_l = box_l
        self.nargp_H17_slice19.box_h = box_h
        self.nargp_H17_slice19.z     = z 
        self.nargp_H17_slice19.slice = slice
        self.nargp_H17_slice19.num_lf = num_lf
        self.nargp_H17_slice19.num_hf = num_hf

        # NARGP: 18 HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60]
        num_hf = 18
        self.nargp_H18_slice19 = ValidationLoader(
            [
                nargp_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.nargp_H18_slice19.res_l = res_l
        self.nargp_H18_slice19.res_h = res_h
        self.nargp_H18_slice19.box_l = box_l
        self.nargp_H18_slice19.box_h = box_h
        self.nargp_H18_slice19.z     = z 
        self.nargp_H18_slice19.slice = slice
        self.nargp_H18_slice19.num_lf = num_lf
        self.nargp_H18_slice19.num_hf = num_hf


        ############################ Various other Boxsizes ############################
        # Box 224
        # dGMGP: Vary redshifts
        res_l = 128
        res_h = 512
        box_l = 256
        box_l_2 = 224
        box_h = 256
        z = [0, 0.2, 0.5, 1.0, 2.0, 3.0] # Forgot to run z=0.5
        # z = [0, 0.2, 1.0, 2.0, 3.0]
        slice = [57, 58, 59]
        num_lf = 60
        num_hf = 3
        self.dgmgp_L60_L2box224_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_L2box224_H3_z0_1_2_slice_19.res_l = res_l
        self.dgmgp_L60_L2box224_H3_z0_1_2_slice_19.res_h = res_h
        self.dgmgp_L60_L2box224_H3_z0_1_2_slice_19.box_l = box_l
        self.dgmgp_L60_L2box224_H3_z0_1_2_slice_19.box_h = box_h
        self.dgmgp_L60_L2box224_H3_z0_1_2_slice_19.z     = z 
        self.dgmgp_L60_L2box224_H3_z0_1_2_slice_19.slice = slice
        self.dgmgp_L60_L2box224_H3_z0_1_2_slice_19.num_lf = num_lf
        self.dgmgp_L60_L2box224_H3_z0_1_2_slice_19.num_hf = num_hf
        # AR1: Vary redshifts
        self.ar1_L60_L2box224_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                ar1_folder_name(num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.ar1_L60_L2box224_H3_z0_1_2_slice_19.res_l = res_l
        self.ar1_L60_L2box224_H3_z0_1_2_slice_19.res_h = res_h
        self.ar1_L60_L2box224_H3_z0_1_2_slice_19.box_l = box_l
        self.ar1_L60_L2box224_H3_z0_1_2_slice_19.box_h = box_h
        self.ar1_L60_L2box224_H3_z0_1_2_slice_19.z     = z 
        self.ar1_L60_L2box224_H3_z0_1_2_slice_19.slice = slice
        self.ar1_L60_L2box224_H3_z0_1_2_slice_19.num_lf = num_lf
        self.ar1_L60_L2box224_H3_z0_1_2_slice_19.num_hf = num_hf
        # NARGP: Vary redshifts
        self.nargp_L60_L2box224_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                nargp_folder_name(num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.nargp_L60_L2box224_H3_z0_1_2_slice_19.res_l = res_l
        self.nargp_L60_L2box224_H3_z0_1_2_slice_19.res_h = res_h
        self.nargp_L60_L2box224_H3_z0_1_2_slice_19.box_l = box_l
        self.nargp_L60_L2box224_H3_z0_1_2_slice_19.box_h = box_h
        self.nargp_L60_L2box224_H3_z0_1_2_slice_19.z     = z 
        self.nargp_L60_L2box224_H3_z0_1_2_slice_19.slice = slice
        self.nargp_L60_L2box224_H3_z0_1_2_slice_19.num_lf = num_lf
        self.nargp_L60_L2box224_H3_z0_1_2_slice_19.num_hf = num_hf


        # Box 192
        # dGMGP: Vary redshifts
        res_l = 128
        res_h = 512
        box_l = 256
        box_l_2 = 192
        box_h = 256
        z = [0, 0.2, 0.5, 1.0, 2.0, 3.0] # Forgot to run z=0.5
        # z = [0, 0.2, 1.0, 2.0, 3.0]
        slice = [57, 58, 59]
        num_lf = 60
        num_hf = 3
        self.dgmgp_L60_L2box192_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_L2box192_H3_z0_1_2_slice_19.res_l = res_l
        self.dgmgp_L60_L2box192_H3_z0_1_2_slice_19.res_h = res_h
        self.dgmgp_L60_L2box192_H3_z0_1_2_slice_19.box_l = box_l
        self.dgmgp_L60_L2box192_H3_z0_1_2_slice_19.box_h = box_h
        self.dgmgp_L60_L2box192_H3_z0_1_2_slice_19.z     = z 
        self.dgmgp_L60_L2box192_H3_z0_1_2_slice_19.slice = slice
        self.dgmgp_L60_L2box192_H3_z0_1_2_slice_19.num_lf = num_lf
        self.dgmgp_L60_L2box192_H3_z0_1_2_slice_19.num_hf = num_hf
        # AR1: Vary redshifts
        self.ar1_L60_L2box192_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                ar1_folder_name(num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.ar1_L60_L2box192_H3_z0_1_2_slice_19.res_l = res_l
        self.ar1_L60_L2box192_H3_z0_1_2_slice_19.res_h = res_h
        self.ar1_L60_L2box192_H3_z0_1_2_slice_19.box_l = box_l
        self.ar1_L60_L2box192_H3_z0_1_2_slice_19.box_h = box_h
        self.ar1_L60_L2box192_H3_z0_1_2_slice_19.z     = z 
        self.ar1_L60_L2box192_H3_z0_1_2_slice_19.slice = slice
        self.ar1_L60_L2box192_H3_z0_1_2_slice_19.num_lf = num_lf
        self.ar1_L60_L2box192_H3_z0_1_2_slice_19.num_hf = num_hf
        # NARGP: Vary redshifts
        self.nargp_L60_L2box192_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                nargp_folder_name(num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.nargp_L60_L2box192_H3_z0_1_2_slice_19.res_l = res_l
        self.nargp_L60_L2box192_H3_z0_1_2_slice_19.res_h = res_h
        self.nargp_L60_L2box192_H3_z0_1_2_slice_19.box_l = box_l
        self.nargp_L60_L2box192_H3_z0_1_2_slice_19.box_h = box_h
        self.nargp_L60_L2box192_H3_z0_1_2_slice_19.z     = z 
        self.nargp_L60_L2box192_H3_z0_1_2_slice_19.slice = slice
        self.nargp_L60_L2box192_H3_z0_1_2_slice_19.num_lf = num_lf
        self.nargp_L60_L2box192_H3_z0_1_2_slice_19.num_hf = num_hf

        # Box 160
        # dGMGP: Vary redshifts
        res_l = 128
        res_h = 512
        box_l = 256
        box_l_2 = 160
        box_h = 256
        z = [0, 0.2, 0.5, 1.0, 2.0, 3.0] # Forgot to run z=0.5
        # z = [0, 0.2, 1.0, 2.0, 3.0]
        slice = [57, 58, 59]
        num_lf = 60
        num_hf = 3
        self.dgmgp_L60_L2box160_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_L2box160_H3_z0_1_2_slice_19.res_l = res_l
        self.dgmgp_L60_L2box160_H3_z0_1_2_slice_19.res_h = res_h
        self.dgmgp_L60_L2box160_H3_z0_1_2_slice_19.box_l = box_l
        self.dgmgp_L60_L2box160_H3_z0_1_2_slice_19.box_h = box_h
        self.dgmgp_L60_L2box160_H3_z0_1_2_slice_19.z     = z 
        self.dgmgp_L60_L2box160_H3_z0_1_2_slice_19.slice = slice
        self.dgmgp_L60_L2box160_H3_z0_1_2_slice_19.num_lf = num_lf
        self.dgmgp_L60_L2box160_H3_z0_1_2_slice_19.num_hf = num_hf
        # AR1: Vary redshifts
        self.ar1_L60_L2box160_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                ar1_folder_name(num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.ar1_L60_L2box160_H3_z0_1_2_slice_19.res_l = res_l
        self.ar1_L60_L2box160_H3_z0_1_2_slice_19.res_h = res_h
        self.ar1_L60_L2box160_H3_z0_1_2_slice_19.box_l = box_l
        self.ar1_L60_L2box160_H3_z0_1_2_slice_19.box_h = box_h
        self.ar1_L60_L2box160_H3_z0_1_2_slice_19.z     = z 
        self.ar1_L60_L2box160_H3_z0_1_2_slice_19.slice = slice
        self.ar1_L60_L2box160_H3_z0_1_2_slice_19.num_lf = num_lf
        self.ar1_L60_L2box160_H3_z0_1_2_slice_19.num_hf = num_hf
        # NARGP: Vary redshifts
        self.nargp_L60_L2box160_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                nargp_folder_name(num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.nargp_L60_L2box160_H3_z0_1_2_slice_19.res_l = res_l
        self.nargp_L60_L2box160_H3_z0_1_2_slice_19.res_h = res_h
        self.nargp_L60_L2box160_H3_z0_1_2_slice_19.box_l = box_l
        self.nargp_L60_L2box160_H3_z0_1_2_slice_19.box_h = box_h
        self.nargp_L60_L2box160_H3_z0_1_2_slice_19.z     = z 
        self.nargp_L60_L2box160_H3_z0_1_2_slice_19.slice = slice
        self.nargp_L60_L2box160_H3_z0_1_2_slice_19.num_lf = num_lf
        self.nargp_L60_L2box160_H3_z0_1_2_slice_19.num_hf = num_hf


        # Box 128
        # dGMGP: Vary redshifts
        res_l = 128
        res_h = 512
        box_l = 256
        box_l_2 = 128
        box_h = 256
        z = [0, 0.2, 0.5, 1.0, 2.0, 3.0] # Forgot to run z=0.5
        # z = [0, 0.2, 1.0, 2.0, 3.0]
        slice = [57, 58, 59]
        num_lf = 60
        num_hf = 3
        self.dgmgp_L60_L2box128_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_L2box128_H3_z0_1_2_slice_19.res_l = res_l
        self.dgmgp_L60_L2box128_H3_z0_1_2_slice_19.res_h = res_h
        self.dgmgp_L60_L2box128_H3_z0_1_2_slice_19.box_l = box_l
        self.dgmgp_L60_L2box128_H3_z0_1_2_slice_19.box_h = box_h
        self.dgmgp_L60_L2box128_H3_z0_1_2_slice_19.z     = z 
        self.dgmgp_L60_L2box128_H3_z0_1_2_slice_19.slice = slice
        self.dgmgp_L60_L2box128_H3_z0_1_2_slice_19.num_lf = num_lf
        self.dgmgp_L60_L2box128_H3_z0_1_2_slice_19.num_hf = num_hf
        # AR1: Vary redshifts
        self.ar1_L60_L2box128_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                ar1_folder_name(num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.ar1_L60_L2box128_H3_z0_1_2_slice_19.res_l = res_l
        self.ar1_L60_L2box128_H3_z0_1_2_slice_19.res_h = res_h
        self.ar1_L60_L2box128_H3_z0_1_2_slice_19.box_l = box_l
        self.ar1_L60_L2box128_H3_z0_1_2_slice_19.box_h = box_h
        self.ar1_L60_L2box128_H3_z0_1_2_slice_19.z     = z 
        self.ar1_L60_L2box128_H3_z0_1_2_slice_19.slice = slice
        self.ar1_L60_L2box128_H3_z0_1_2_slice_19.num_lf = num_lf
        self.ar1_L60_L2box128_H3_z0_1_2_slice_19.num_hf = num_hf
        # NARGP: Vary redshifts
        self.nargp_L60_L2box128_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                nargp_folder_name(num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.nargp_L60_L2box128_H3_z0_1_2_slice_19.res_l = res_l
        self.nargp_L60_L2box128_H3_z0_1_2_slice_19.res_h = res_h
        self.nargp_L60_L2box128_H3_z0_1_2_slice_19.box_l = box_l
        self.nargp_L60_L2box128_H3_z0_1_2_slice_19.box_h = box_h
        self.nargp_L60_L2box128_H3_z0_1_2_slice_19.z     = z 
        self.nargp_L60_L2box128_H3_z0_1_2_slice_19.slice = slice
        self.nargp_L60_L2box128_H3_z0_1_2_slice_19.num_lf = num_lf
        self.nargp_L60_L2box128_H3_z0_1_2_slice_19.num_hf = num_hf

        # Box 100
        res_l = 128
        res_h = 512
        box_l = 256
        box_l_2 = 128
        box_h = 256
        z = [0, 0.2, 0.5, 1.0, 2.0, 3.0] # Forgot to run z=0.5
        # z = [0, 0.2, 1.0, 2.0, 3.0]
        slice = [57, 58, 59]
        num_lf = 60
        num_hf = 3
        # AR1: Vary redshifts
        self.ar1_L60_L2box100_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                ar1_folder_name(num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.ar1_L60_L2box100_H3_z0_1_2_slice_19.res_l = res_l
        self.ar1_L60_L2box100_H3_z0_1_2_slice_19.res_h = res_h
        self.ar1_L60_L2box100_H3_z0_1_2_slice_19.box_l = box_l
        self.ar1_L60_L2box100_H3_z0_1_2_slice_19.box_h = box_h
        self.ar1_L60_L2box100_H3_z0_1_2_slice_19.z     = z 
        self.ar1_L60_L2box100_H3_z0_1_2_slice_19.slice = slice
        self.ar1_L60_L2box100_H3_z0_1_2_slice_19.num_lf = num_lf
        self.ar1_L60_L2box100_H3_z0_1_2_slice_19.num_hf = num_hf
        # NARGP: Vary redshifts
        self.nargp_L60_L2box100_H3_z0_1_2_slice_19 = ValidationLoader(
            [
                nargp_folder_name(num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.nargp_L60_L2box100_H3_z0_1_2_slice_19.res_l = res_l
        self.nargp_L60_L2box100_H3_z0_1_2_slice_19.res_h = res_h
        self.nargp_L60_L2box100_H3_z0_1_2_slice_19.box_l = box_l
        self.nargp_L60_L2box100_H3_z0_1_2_slice_19.box_h = box_h
        self.nargp_L60_L2box100_H3_z0_1_2_slice_19.z     = z 
        self.nargp_L60_L2box100_H3_z0_1_2_slice_19.slice = slice
        self.nargp_L60_L2box100_H3_z0_1_2_slice_19.num_lf = num_lf
        self.nargp_L60_L2box100_H3_z0_1_2_slice_19.num_hf = num_hf


        ############################ 360 training data ############################
        # AR1: vary LF - to 360
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [57, 58, 59]
        num_lf = [12, 18, 24, 30, 36, 42, 48, 54, 60, 70, 80, 90, 100, 150, 200, 250, 300, 360]
        num_hf = 3
        self.ar1_H3_slice19_added_360 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, num_hf, res_h, box_h, z, slice) for n_lf in num_lf
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H3_slice19_added_360.res_l = res_l
        self.ar1_H3_slice19_added_360.res_h = res_h
        self.ar1_H3_slice19_added_360.box_l = box_l
        self.ar1_H3_slice19_added_360.box_h = box_h
        self.ar1_H3_slice19_added_360.z     = z 
        self.ar1_H3_slice19_added_360.slice = slice
        self.ar1_H3_slice19_added_360.num_lf = num_lf
        self.ar1_H3_slice19_added_360.num_hf = num_hf


        ############################ SLHD versus not using slices ############################
        # 5, 41, 57,
        # xx 39, 15, 8,
        # 7, 11, 4,
        # 3, 59, 12,
        # xx 16, 9, 14,
        # 40, 0, 1,
        # xx 10, 17, 6,
        # 58, 2, 13
        ############################ Various slices ############################
        # Slice: 5, 41, 57,
        # # dGMGP: Vary redshifts
        res_l = 128
        res_h = 512
        box_l = 256
        box_l_2 = 100
        box_h = 256
        z = [0, 0.2, 0.5, 1.0, 2.0, 3.0]
        slice = [5, 41, 57]
        num_lf = 60
        num_hf = 3
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_5_41_57 = ValidationLoader(
            [
                dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_5_41_57.res_l = res_l
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_5_41_57.res_h = res_h
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_5_41_57.box_l = box_l
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_5_41_57.box_h = box_h
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_5_41_57.z     = z 
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_5_41_57.slice = slice
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_5_41_57.num_lf = num_lf
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_5_41_57.num_hf = num_hf
        # AR1: Vary redshifts
        self.ar1_L60_L2box100_H3_z0_1_2_slice_5_41_57 = ValidationLoader(
            [
                ar1_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.ar1_L60_L2box100_H3_z0_1_2_slice_5_41_57.res_l = res_l
        self.ar1_L60_L2box100_H3_z0_1_2_slice_5_41_57.res_h = res_h
        self.ar1_L60_L2box100_H3_z0_1_2_slice_5_41_57.box_l = box_l
        self.ar1_L60_L2box100_H3_z0_1_2_slice_5_41_57.box_h = box_h
        self.ar1_L60_L2box100_H3_z0_1_2_slice_5_41_57.z     = z 
        self.ar1_L60_L2box100_H3_z0_1_2_slice_5_41_57.slice = slice
        self.ar1_L60_L2box100_H3_z0_1_2_slice_5_41_57.num_lf = num_lf
        self.ar1_L60_L2box100_H3_z0_1_2_slice_5_41_57.num_hf = num_hf
        # NARGP: Vary redshifts
        self.nargp_L60_L2box100_H3_z0_1_2_slice_5_41_57 = ValidationLoader(
            [
                nargp_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.nargp_L60_L2box100_H3_z0_1_2_slice_5_41_57.res_l = res_l
        self.nargp_L60_L2box100_H3_z0_1_2_slice_5_41_57.res_h = res_h
        self.nargp_L60_L2box100_H3_z0_1_2_slice_5_41_57.box_l = box_l
        self.nargp_L60_L2box100_H3_z0_1_2_slice_5_41_57.box_h = box_h
        self.nargp_L60_L2box100_H3_z0_1_2_slice_5_41_57.z     = z 
        self.nargp_L60_L2box100_H3_z0_1_2_slice_5_41_57.slice = slice
        self.nargp_L60_L2box100_H3_z0_1_2_slice_5_41_57.num_lf = num_lf
        self.nargp_L60_L2box100_H3_z0_1_2_slice_5_41_57.num_hf = num_hf

        # # Slice: 8, 15, 39
        # # # dGMGP: Vary redshifts
        # res_l = 128
        # res_h = 512
        # box_l = 256
        # box_l_2 = 100
        # box_h = 256
        # z = [0, 0.2, 0.5, 1.0, 2.0, 3.0]
        # slice = [8, 15, 39]
        # num_lf = 60
        # num_hf = 3
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_8_15_39 = ValidationLoader(
        #     [
        #         dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
        #     ],
        #     num_lowres_list=[num_lf for _ in z],
        #     num_highres=num_hf,
        #     dGMGP=True,
        # )
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_8_15_39.res_l = res_l
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_8_15_39.res_h = res_h
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_8_15_39.box_l = box_l
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_8_15_39.box_h = box_h
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_8_15_39.z     = z 
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_8_15_39.slice = slice
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_8_15_39.num_lf = num_lf
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_8_15_39.num_hf = num_hf
        # # AR1: Vary redshifts
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_8_15_39 = ValidationLoader(
        #     [
        #         ar1_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
        #     ],
        #     num_lowres_list=[num_lf for _ in z],
        #     num_highres=num_hf,
        # )
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_8_15_39.res_l = res_l
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_8_15_39.res_h = res_h
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_8_15_39.box_l = box_l
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_8_15_39.box_h = box_h
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_8_15_39.z     = z 
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_8_15_39.slice = slice
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_8_15_39.num_lf = num_lf
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_8_15_39.num_hf = num_hf
        # # NARGP: Vary redshifts
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_8_15_39 = ValidationLoader(
        #     [
        #         nargp_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
        #     ],
        #     num_lowres_list=[num_lf for _ in z],
        #     num_highres=num_hf,
        # )
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_8_15_39.res_l = res_l
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_8_15_39.res_h = res_h
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_8_15_39.box_l = box_l
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_8_15_39.box_h = box_h
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_8_15_39.z     = z 
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_8_15_39.slice = slice
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_8_15_39.num_lf = num_lf
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_8_15_39.num_hf = num_hf

        # Slice: 4, 7, 11
        # # dGMGP: Vary redshifts
        res_l = 128
        res_h = 512
        box_l = 256
        box_l_2 = 100
        box_h = 256
        z = [0, 0.2, 0.5, 1.0, 2.0, 3.0]
        slice = [4, 7, 11]
        num_lf = 60
        num_hf = 3
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_4_7_11 = ValidationLoader(
            [
                dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_4_7_11.res_l = res_l
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_4_7_11.res_h = res_h
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_4_7_11.box_l = box_l
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_4_7_11.box_h = box_h
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_4_7_11.z     = z 
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_4_7_11.slice = slice
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_4_7_11.num_lf = num_lf
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_4_7_11.num_hf = num_hf
        # AR1: Vary redshifts
        self.ar1_L60_L2box100_H3_z0_1_2_slice_4_7_11 = ValidationLoader(
            [
                ar1_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.ar1_L60_L2box100_H3_z0_1_2_slice_4_7_11.res_l = res_l
        self.ar1_L60_L2box100_H3_z0_1_2_slice_4_7_11.res_h = res_h
        self.ar1_L60_L2box100_H3_z0_1_2_slice_4_7_11.box_l = box_l
        self.ar1_L60_L2box100_H3_z0_1_2_slice_4_7_11.box_h = box_h
        self.ar1_L60_L2box100_H3_z0_1_2_slice_4_7_11.z     = z 
        self.ar1_L60_L2box100_H3_z0_1_2_slice_4_7_11.slice = slice
        self.ar1_L60_L2box100_H3_z0_1_2_slice_4_7_11.num_lf = num_lf
        self.ar1_L60_L2box100_H3_z0_1_2_slice_4_7_11.num_hf = num_hf
        # NARGP: Vary redshifts
        self.nargp_L60_L2box100_H3_z0_1_2_slice_4_7_11 = ValidationLoader(
            [
                nargp_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.nargp_L60_L2box100_H3_z0_1_2_slice_4_7_11.res_l = res_l
        self.nargp_L60_L2box100_H3_z0_1_2_slice_4_7_11.res_h = res_h
        self.nargp_L60_L2box100_H3_z0_1_2_slice_4_7_11.box_l = box_l
        self.nargp_L60_L2box100_H3_z0_1_2_slice_4_7_11.box_h = box_h
        self.nargp_L60_L2box100_H3_z0_1_2_slice_4_7_11.z     = z 
        self.nargp_L60_L2box100_H3_z0_1_2_slice_4_7_11.slice = slice
        self.nargp_L60_L2box100_H3_z0_1_2_slice_4_7_11.num_lf = num_lf
        self.nargp_L60_L2box100_H3_z0_1_2_slice_4_7_11.num_hf = num_hf


        # Slice: 3, 12, 59 
        # # dGMGP: Vary redshifts
        res_l = 128
        res_h = 512
        box_l = 256
        box_l_2 = 100
        box_h = 256
        z = [0, 0.2, 0.5, 1.0, 2.0, 3.0]
        slice = [3, 12, 59]
        num_lf = 60
        num_hf = 3
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_3_12_59 = ValidationLoader(
            [
                dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_3_12_59.res_l = res_l
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_3_12_59.res_h = res_h
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_3_12_59.box_l = box_l
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_3_12_59.box_h = box_h
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_3_12_59.z     = z 
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_3_12_59.slice = slice
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_3_12_59.num_lf = num_lf
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_3_12_59.num_hf = num_hf
        # AR1: Vary redshifts
        self.ar1_L60_L2box100_H3_z0_1_2_slice_3_12_59 = ValidationLoader(
            [
                ar1_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.ar1_L60_L2box100_H3_z0_1_2_slice_3_12_59.res_l = res_l
        self.ar1_L60_L2box100_H3_z0_1_2_slice_3_12_59.res_h = res_h
        self.ar1_L60_L2box100_H3_z0_1_2_slice_3_12_59.box_l = box_l
        self.ar1_L60_L2box100_H3_z0_1_2_slice_3_12_59.box_h = box_h
        self.ar1_L60_L2box100_H3_z0_1_2_slice_3_12_59.z     = z 
        self.ar1_L60_L2box100_H3_z0_1_2_slice_3_12_59.slice = slice
        self.ar1_L60_L2box100_H3_z0_1_2_slice_3_12_59.num_lf = num_lf
        self.ar1_L60_L2box100_H3_z0_1_2_slice_3_12_59.num_hf = num_hf
        # NARGP: Vary redshifts
        self.nargp_L60_L2box100_H3_z0_1_2_slice_3_12_59 = ValidationLoader(
            [
                nargp_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.nargp_L60_L2box100_H3_z0_1_2_slice_3_12_59.res_l = res_l
        self.nargp_L60_L2box100_H3_z0_1_2_slice_3_12_59.res_h = res_h
        self.nargp_L60_L2box100_H3_z0_1_2_slice_3_12_59.box_l = box_l
        self.nargp_L60_L2box100_H3_z0_1_2_slice_3_12_59.box_h = box_h
        self.nargp_L60_L2box100_H3_z0_1_2_slice_3_12_59.z     = z 
        self.nargp_L60_L2box100_H3_z0_1_2_slice_3_12_59.slice = slice
        self.nargp_L60_L2box100_H3_z0_1_2_slice_3_12_59.num_lf = num_lf
        self.nargp_L60_L2box100_H3_z0_1_2_slice_3_12_59.num_hf = num_hf


        # # Slice: 9, 14, 16
        # # # dGMGP: Vary redshifts
        # res_l = 128
        # res_h = 512
        # box_l = 256
        # box_l_2 = 100
        # box_h = 256
        # z = [0, 0.2, 0.5, 1.0, 2.0, 3.0]
        # slice = [9, 14, 16]
        # num_lf = 60
        # num_hf = 3
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_9_14_16 = ValidationLoader(
        #     [
        #         dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
        #     ],
        #     num_lowres_list=[num_lf for _ in z],
        #     num_highres=num_hf,
        #     dGMGP=True,
        # )
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_9_14_16.res_l = res_l
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_9_14_16.res_h = res_h
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_9_14_16.box_l = box_l
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_9_14_16.box_h = box_h
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_9_14_16.z     = z 
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_9_14_16.slice = slice
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_9_14_16.num_lf = num_lf
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_9_14_16.num_hf = num_hf
        # # AR1: Vary redshifts
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_9_14_16 = ValidationLoader(
        #     [
        #         ar1_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
        #     ],
        #     num_lowres_list=[num_lf for _ in z],
        #     num_highres=num_hf,
        # )
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_9_14_16.res_l = res_l
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_9_14_16.res_h = res_h
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_9_14_16.box_l = box_l
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_9_14_16.box_h = box_h
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_9_14_16.z     = z 
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_9_14_16.slice = slice
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_9_14_16.num_lf = num_lf
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_9_14_16.num_hf = num_hf
        # # NARGP: Vary redshifts
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_9_14_16 = ValidationLoader(
        #     [
        #         nargp_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
        #     ],
        #     num_lowres_list=[num_lf for _ in z],
        #     num_highres=num_hf,
        # )
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_9_14_16.res_l = res_l
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_9_14_16.res_h = res_h
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_9_14_16.box_l = box_l
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_9_14_16.box_h = box_h
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_9_14_16.z     = z 
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_9_14_16.slice = slice
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_9_14_16.num_lf = num_lf
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_9_14_16.num_hf = num_hf


        # Slice: 0, 1, 40
        # # dGMGP: Vary redshifts
        res_l = 128
        res_h = 512
        box_l = 256
        box_l_2 = 100
        box_h = 256
        z = [0, 0.2, 0.5, 1.0, 2.0, 3.0]
        slice = [0, 1, 40]
        num_lf = 60
        num_hf = 3
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_0_1_40 = ValidationLoader(
            [
                dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_0_1_40.res_l = res_l
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_0_1_40.res_h = res_h
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_0_1_40.box_l = box_l
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_0_1_40.box_h = box_h
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_0_1_40.z     = z 
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_0_1_40.slice = slice
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_0_1_40.num_lf = num_lf
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_0_1_40.num_hf = num_hf
        # AR1: Vary redshifts
        self.ar1_L60_L2box100_H3_z0_1_2_slice_0_1_40 = ValidationLoader(
            [
                ar1_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.ar1_L60_L2box100_H3_z0_1_2_slice_0_1_40.res_l = res_l
        self.ar1_L60_L2box100_H3_z0_1_2_slice_0_1_40.res_h = res_h
        self.ar1_L60_L2box100_H3_z0_1_2_slice_0_1_40.box_l = box_l
        self.ar1_L60_L2box100_H3_z0_1_2_slice_0_1_40.box_h = box_h
        self.ar1_L60_L2box100_H3_z0_1_2_slice_0_1_40.z     = z 
        self.ar1_L60_L2box100_H3_z0_1_2_slice_0_1_40.slice = slice
        self.ar1_L60_L2box100_H3_z0_1_2_slice_0_1_40.num_lf = num_lf
        self.ar1_L60_L2box100_H3_z0_1_2_slice_0_1_40.num_hf = num_hf
        # NARGP: Vary redshifts
        self.nargp_L60_L2box100_H3_z0_1_2_slice_0_1_40 = ValidationLoader(
            [
                nargp_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.nargp_L60_L2box100_H3_z0_1_2_slice_0_1_40.res_l = res_l
        self.nargp_L60_L2box100_H3_z0_1_2_slice_0_1_40.res_h = res_h
        self.nargp_L60_L2box100_H3_z0_1_2_slice_0_1_40.box_l = box_l
        self.nargp_L60_L2box100_H3_z0_1_2_slice_0_1_40.box_h = box_h
        self.nargp_L60_L2box100_H3_z0_1_2_slice_0_1_40.z     = z 
        self.nargp_L60_L2box100_H3_z0_1_2_slice_0_1_40.slice = slice
        self.nargp_L60_L2box100_H3_z0_1_2_slice_0_1_40.num_lf = num_lf
        self.nargp_L60_L2box100_H3_z0_1_2_slice_0_1_40.num_hf = num_hf


        # # Slice: 6, 10, 17
        # # # dGMGP: Vary redshifts
        # res_l = 128
        # res_h = 512
        # box_l = 256
        # box_l_2 = 100
        # box_h = 256
        # z = [0, 0.2, 0.5, 1.0, 2.0, 3.0]
        # slice = [6, 10, 17]
        # num_lf = 60
        # num_hf = 3
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_6_10_17 = ValidationLoader(
        #     [
        #         dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
        #     ],
        #     num_lowres_list=[num_lf for _ in z],
        #     num_highres=num_hf,
        #     dGMGP=True,
        # )
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_6_10_17.res_l = res_l
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_6_10_17.res_h = res_h
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_6_10_17.box_l = box_l
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_6_10_17.box_h = box_h
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_6_10_17.z     = z 
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_6_10_17.slice = slice
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_6_10_17.num_lf = num_lf
        # self.dgmgp_L60_L2box100_H3_z0_1_2_slice_6_10_17.num_hf = num_hf
        # # AR1: Vary redshifts
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_6_10_17 = ValidationLoader(
        #     [
        #         ar1_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
        #     ],
        #     num_lowres_list=[num_lf for _ in z],
        #     num_highres=num_hf,
        # )
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_6_10_17.res_l = res_l
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_6_10_17.res_h = res_h
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_6_10_17.box_l = box_l
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_6_10_17.box_h = box_h
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_6_10_17.z     = z 
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_6_10_17.slice = slice
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_6_10_17.num_lf = num_lf
        # self.ar1_L60_L2box100_H3_z0_1_2_slice_6_10_17.num_hf = num_hf
        # # NARGP: Vary redshifts
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_6_10_17 = ValidationLoader(
        #     [
        #         nargp_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
        #     ],
        #     num_lowres_list=[num_lf for _ in z],
        #     num_highres=num_hf,
        # )
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_6_10_17.res_l = res_l
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_6_10_17.res_h = res_h
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_6_10_17.box_l = box_l
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_6_10_17.box_h = box_h
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_6_10_17.z     = z 
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_6_10_17.slice = slice
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_6_10_17.num_lf = num_lf
        # self.nargp_L60_L2box100_H3_z0_1_2_slice_6_10_17.num_hf = num_hf


        # Slice: 2, 13, 58
        # # dGMGP: Vary redshifts
        res_l = 128
        res_h = 512
        box_l = 256
        box_l_2 = 100
        box_h = 256
        z = [0, 0.2, 0.5, 1.0, 2.0, 3.0]
        slice = [2, 13, 58]
        num_lf = 60
        num_hf = 3
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_2_13_58 = ValidationLoader(
            [
                dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
            dGMGP=True,
        )
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_2_13_58.res_l = res_l
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_2_13_58.res_h = res_h
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_2_13_58.box_l = box_l
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_2_13_58.box_h = box_h
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_2_13_58.z     = z 
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_2_13_58.slice = slice
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_2_13_58.num_lf = num_lf
        self.dgmgp_L60_L2box100_H3_z0_1_2_slice_2_13_58.num_hf = num_hf
        # AR1: Vary redshifts
        self.ar1_L60_L2box100_H3_z0_1_2_slice_2_13_58 = ValidationLoader(
            [
                ar1_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.ar1_L60_L2box100_H3_z0_1_2_slice_2_13_58.res_l = res_l
        self.ar1_L60_L2box100_H3_z0_1_2_slice_2_13_58.res_h = res_h
        self.ar1_L60_L2box100_H3_z0_1_2_slice_2_13_58.box_l = box_l
        self.ar1_L60_L2box100_H3_z0_1_2_slice_2_13_58.box_h = box_h
        self.ar1_L60_L2box100_H3_z0_1_2_slice_2_13_58.z     = z 
        self.ar1_L60_L2box100_H3_z0_1_2_slice_2_13_58.slice = slice
        self.ar1_L60_L2box100_H3_z0_1_2_slice_2_13_58.num_lf = num_lf
        self.ar1_L60_L2box100_H3_z0_1_2_slice_2_13_58.num_hf = num_hf
        # NARGP: Vary redshifts
        self.nargp_L60_L2box100_H3_z0_1_2_slice_2_13_58 = ValidationLoader(
            [
                nargp_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
            ],
            num_lowres_list=[num_lf for _ in z],
            num_highres=num_hf,
        )
        self.nargp_L60_L2box100_H3_z0_1_2_slice_2_13_58.res_l = res_l
        self.nargp_L60_L2box100_H3_z0_1_2_slice_2_13_58.res_h = res_h
        self.nargp_L60_L2box100_H3_z0_1_2_slice_2_13_58.box_l = box_l
        self.nargp_L60_L2box100_H3_z0_1_2_slice_2_13_58.box_h = box_h
        self.nargp_L60_L2box100_H3_z0_1_2_slice_2_13_58.z     = z 
        self.nargp_L60_L2box100_H3_z0_1_2_slice_2_13_58.slice = slice
        self.nargp_L60_L2box100_H3_z0_1_2_slice_2_13_58.num_lf = num_lf
        self.nargp_L60_L2box100_H3_z0_1_2_slice_2_13_58.num_hf = num_hf


        ############################ SLHD slices ############################
        all_slices = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10 ,11],
            [12, 13, 14],
            [39, 40, 41],
            [57, 58, 59],
        ]
        for slice in all_slices:
            # # dGMGP: Vary redshifts
            res_l = 128
            res_h = 512
            box_l = 256
            box_l_2 = 100
            box_h = 256
            z = [0, 0.2, 0.5, 1.0, 2.0, 3.0]
            # slice = [5, 41, 57]
            num_lf = 60
            num_hf = 3
            setattr(self, "dgmgp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice))), ValidationLoader(
                    [
                        dgmgp_folder_name(num_lf, res_l, box_l, num_lf, res_l, box_l_2, num_hf, res_h, box_h, zz, slice) for zz in z
                    ],
                    num_lowres_list=[num_lf for _ in z],
                    num_highres=num_hf,
                    dGMGP=True,
                )
            )
            setattr(getattr(self, "dgmgp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "res_l", res_l)
            setattr(getattr(self, "dgmgp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "res_h", res_h)
            setattr(getattr(self, "dgmgp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "box_l", box_l)
            setattr(getattr(self, "dgmgp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "box_h", box_h)
            setattr(getattr(self, "dgmgp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "z"    , z)
            setattr(getattr(self, "dgmgp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "slice", slice)
            setattr(getattr(self, "dgmgp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "num_lf", num_lf)
            setattr(getattr(self, "dgmgp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "num_hf", num_hf)
            # AR1: Vary redshifts
            setattr(self, "ar1_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice))), ValidationLoader(
                    [
                        ar1_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
                    ],
                    num_lowres_list=[num_lf for _ in z],
                    num_highres=num_hf,
                )
            )
            setattr(getattr(self, "ar1_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "res_l", res_l)
            setattr(getattr(self, "ar1_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "res_h", res_h)
            setattr(getattr(self, "ar1_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "box_l", box_l)
            setattr(getattr(self, "ar1_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "box_h", box_h)
            setattr(getattr(self, "ar1_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "z",     z )
            setattr(getattr(self, "ar1_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "slice", slice)
            setattr(getattr(self, "ar1_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "num_lf", num_lf)
            setattr(getattr(self, "ar1_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "num_hf", num_hf)
            # NARGP: Vary redshifts
            setattr(self, "nargp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice))), ValidationLoader(
                    [
                        nargp_folder_name(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
                    ],
                    num_lowres_list=[num_lf for _ in z],
                    num_highres=num_hf,
                )
            )
            setattr(getattr(self, "nargp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "res_l", res_l)
            setattr(getattr(self, "nargp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "res_h", res_h)
            setattr(getattr(self, "nargp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "box_l", box_l)
            setattr(getattr(self, "nargp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "box_h", box_h)
            setattr(getattr(self, "nargp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "z"    , z)
            setattr(getattr(self, "nargp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "slice", slice)
            setattr(getattr(self, "nargp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "num_lf", num_lf)
            setattr(getattr(self, "nargp_L60_L2box100_H3_z0_1_2_slice_{}".format("_".join(map(str, slice)))), "num_hf", num_hf)

        # change back to original dir
        os.chdir(old_dir)

def get_mean_std(vloader: ValidationLoader, ):
    """
    A modification from Martin's plotting script
    """
    absmean = np.mean(vloader.relative_errors, axis=2)
    absstd  = np.std( vloader.relative_errors, axis=2)

    absmeanHF = np.mean(vloader.relative_errors_hf, axis=2)
    absstdHF  = np.std( vloader.relative_errors_hf, axis=2)

    return absmean, absstd, absmeanHF, absstdHF

# get error for a specific emulator (averaged over z or k)
def get_mean_std_one(lf: Optional[int], hf: Optional[int], z_or_k: str = 'k', nargp: bool = False):
    """
    Modification from Martin's function

    Parameters:
    ----
    lf : number of LF points
    HF : number of HF points
    z_or_k : errors as function of z or k, i.e., average over k or z
    nargp : use nargp results or ar1 results
    """
    axis_z = 0
    axis_test_cases = 1
    axis_k = 2

    if not nargp:
        folder_fn = ar1_folder_name
    elif nargp:
        folder_fn = nargp_folder_name

    # TODO: fix these for now
    res_l = 128
    res_h = 512
    box_l = 256
    box_h = 256
    z = [0, 0.2, 0.5, 1.0, 2.0, 3.0]
    slice = [57, 58, 59]
    num_lf = lf
    num_hf = hf
    
    # list of emulators in different redshifts
    vloader = ValidationLoader(
        [
            folder_fn(num_lf, res_l, box_l, num_hf, res_h, box_h, zz, slice) for zz in z
        ],
        num_lowres_list=[num_lf for _ in z],
        num_highres=num_hf,
    )
    vloader.res_l = res_l
    vloader.res_h = res_h
    vloader.box_l = box_l
    vloader.box_h = box_h
    vloader.z     = z 
    vloader.slice = slice
    vloader.num_lf = num_lf
    vloader.num_hf = num_hf

    # average over k
    # pred_div_exact: Shape (number of emulators (redshifts in this case), number of test simulations, number of k bins)
    if z_or_k == 'z':
        absmean = np.full(absmean.shape[axis_z], fill_value=np.nan)
        absstd  = np.full(absmean.shape[axis_z], fill_value=np.nan)
        for i in enumerate(absmean):
            absmean[i] = np.mean(np.abs(vloader.pred_div_exact - 1)[i, :, :])
            absstd[i]  = np.std( np.abs(vloader.pred_div_exact - 1)[i, :, :])

    elif z_or_k == 'k':
        absmean = np.full(absmean.shape[axis_k], fill_value=np.nan)
        absstd  = np.full(absmean.shape[axis_k], fill_value=np.nan)
        for i in enumerate(absmean):
            absmean[i] = np.mean(np.abs(vloader.pred_div_exact - 1)[:, :, i])
            absstd[i]  = np.std( np.abs(vloader.pred_div_exact - 1)[:, :, i])

    return vloader, absmean, absstd

from matter_multi_fidelity_emu.data_loader_dgmgp import interpolate

def interp_lf_trim_hf(k_lf: np.ndarray, k_hf: np.ndarray, Y_lf: np.ndarray, Y_hf: np.ndarray):
    # Interpolate the LF,
    # and limit the HF kmax to the maximum of LF
    # Min k bins LF <= k bins HF <= Max k bins LF
    ind_min = (np.log10(k_lf).min() <= np.log10(k_hf)) & (
        np.log10(k_hf) <= np.log10(k_lf).max()
    )

    # interpolate: interp(log10_k, Y_lf)(log10_k[ind_min])
    # I do want to interpolate in loglog scale.
    # TODO: Think about if our smooth prior is on linear or log scale?
    Y_lf_new = interpolate(
        np.log10(k_lf), np.log10(Y_lf), np.log10(k_hf)[ind_min]
    )
    Y_lf = 10**Y_lf_new
    Y_hf = Y_hf[:, ind_min]
    k_hf = k_hf[ind_min]
    k_lf = k_hf

    return k_lf, k_hf, Y_lf, Y_hf, ind_min

