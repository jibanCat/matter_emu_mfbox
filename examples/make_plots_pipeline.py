from curses import KEY_BEG
import os
from typing import Optional
import numpy as np

from matplotlib import pyplot as plt
from regex import D

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

        # AR1: vary LF
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


        # dGMGP: vary LF
        res_l = 128
        res_h = 512
        box_l_1 = 256
        box_l_2 = 100
        box_h = 256
        z = 0
        slice = [57, 58, 59]
        num_lf = [18, 24, 30, 36, 42, 48, 54, 60]
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

        # dGMGP: vary LF
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [57, 58, 59]
        num_lf = [18, 24, 30, 36, 42, 48, 54, 60] # assume L1 and L2 have the same L points
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


        # 3HR-6HR-9HR
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        z = 0
        slice = [[57, 58, 59], [0, 1, 2, 57, 58, 59], [0, 1, 2, 3, 4, 5, 57, 58, 59]]
        num_lf = [60, 60, 60]
        num_hf = [3, 6, 9]
        self.ar1_H3_6_9 = ValidationLoader(
            [
                ar1_folder_name(n_lf, res_l, box_l, n_hf, res_h, box_h, z, ss) for n_lf, n_hf, ss in zip(num_lf, num_hf, slice)
            ],
            num_lowres_list=num_lf,
            num_highres=num_hf,
        )
        self.ar1_H3_6_9.res_l = res_l
        self.ar1_H3_6_9.res_h = res_h
        self.ar1_H3_6_9.box_l = box_l
        self.ar1_H3_6_9.box_h = box_h
        self.ar1_H3_6_9.z     = z 
        self.ar1_H3_6_9.slice = slice
        self.ar1_H3_6_9.num_lf = num_lf
        self.ar1_H3_6_9.num_hf = num_hf

        # AR1: Vary redshifts
        res_l = 128
        res_h = 512
        box_l = 256
        box_h = 256
        # z = [0, 0.2, 0.5, 1.0, 2.0, 3.0] # Forgot to run z=0.5
        z = [0, 0.2, 1.0, 2.0, 3.0]
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
        # z = [0, 0.2, 0.5, 1.0, 2.0, 3.0] # Forgot to run z=0.5
        z = [0, 0.2, 1.0, 2.0, 3.0]
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
        # z = [0, 0.2, 0.5, 1.0, 2.0, 3.0] # Forgot to run z=0.5
        z = [0, 0.2, 1.0, 2.0, 3.0]
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

