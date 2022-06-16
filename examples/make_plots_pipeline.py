import os
import numpy as np

from matplotlib import pyplot as plt

from matter_multi_fidelity_emu.plottings.validation_loader import ValidationLoader
from matter_multi_fidelity_emu.data_loader import folder_name
from .make_validations_dgmgp import folder_name as dgmgp_folder_name

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



class PreloadedVloaders:
    """
    Prepare validation loaders
    """

    def __init__(self, img_dir: str = "data/output/"):
        old_dir = os.getcwd()
        os.chdir(img_dir)

        # 3HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [57, 58, 59]
        num_lf = [18, 24, 30, 36, 42, 48, 54, 60]
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


        # 3HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [57, 58, 59]
        num_lf = [18, 24, 30, 36, 42, 48, 54, 60]
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

        # change back to original dir
        os.chdir(old_dir)
