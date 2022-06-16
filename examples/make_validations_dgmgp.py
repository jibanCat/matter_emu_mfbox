from typing import List

import os
import json

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from matter_multi_fidelity_emu.gpemulator_singlebin import (
    SingleBinGP,
    SingleBinLinearGP,
    SingleBinNonLinearGP,
)
from matter_multi_fidelity_emu.data_loader import PowerSpecs

from matter_multi_fidelity_emu.gpemulator_singlebin import SingleBindGMGP

from matter_multi_fidelity_emu.data_loader_dgmgp import interpolate

# set a random number seed to reproducibility
np.random.seed(0)

matplotlib.use("pdf")

n_save = 10 # shrink power spec; for testing purpose

save_figure = lambda filename: plt.savefig(
    "{}.pdf".format(filename), format="pdf", dpi=300
)

def folder_name(
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
    return "dGMGP_{}_L1-res{}box{}_{}_L2-res{}box{}_{}_H-res{}box{}_z{}_ind-{}".format(
        num1,
        res1,
        box1,
        num2,
        res2,
        box2,
        num3,
        res3,
        box3,
        "{:.2g}".format(z).replace(".", "_"),
        "-".join(map(str, selected_ind)),
    )


def generate_data(
        folder_1: str = "data/50_LR_3_HR_box_512_combined_128res",
        folder_2: str = "data/50_LR_box_300_3_HR_box_512_combined_128res",
        n_fidelities: int = 2,
    ):
    data_1 = PowerSpecs(n_fidelities=n_fidelities,)
    data_1.read_from_txt(folder=folder_1)

    data_2 = PowerSpecs(n_fidelities=n_fidelities,)
    data_2.read_from_txt(folder=folder_2)

    return data_1, data_2



def do_validations(
    folder_1: str = "data/50_LR_3_HR_box_512_combined_128res",
    folder_2: str = "data/50_LR_box_300_3_HR_box_512_combined_128res",
    n_optimization_restarts: int = 30,
    n_fidelities: int = 2,
    output_folder: str = "output/dGMGP_50_dmonly128_m1mpc512_m2mpc300_3_dmonly512",
    ARD_last_fidelity: bool = False,
):
    """
    Train and test models, and plot
    1. predicted / exact power spectrum
    2. absolute error plot
    3. parameter plots

    Only support 2 fidelities now.

    Parameters:
    ----
    folder: the folder contains the the training and testing data. See data/50_LR_3_HR
        for example.
    n_optimization_restarts: number of optimization you want to repeat. The GPy will
        choose the best hyperparameters among those repetitions. More is better.
    n_fidelities: only supports 2 now. You may try a larger number but some tweaks might
        be needed.
    turn_off_bias_nargp: not adding bias kernel for NARGP in high-fidelity. In case you
        find the optimization result is not stable, try turning off bias kernel. Some time
        the training data at high-fidelity is not enough to train the bias kernel and
        induce some unstable predictions.
    ARD_last_fidelity: whether to apply ARD for the last (highest) fidelity.
        Default, False.
    """
    # create output folder, recursively
    os.makedirs(output_folder, exist_ok=True)
    old_dir = os.getcwd()
    print("Current path:", old_dir)

    # get training and testing data. Normalization included.
    data_1, data_2 = generate_data(folder_1=folder_1, folder_2=folder_2)

    # highres: log10_ks; lowres: log10_k
    # we are emulating data_1's highres
    log10_k_target = data_1.kf
    log10_k_train  = data_2.kf
    ind_min = (log10_k_target >= log10_k_train.min()) & (log10_k_target <= log10_k_train.max())

    # interpolate: interp(log10_k, Y_lf)(log10_k[ind_min])
    Y_lf_norm_2 = interpolate(data_2.kf, data_2.Y_train_norm[0], data_1.kf[ind_min])
    Y_lf_2 = interpolate(data_2.kf, data_2.Y_train[0], data_1.kf[ind_min])
    assert Y_lf_norm_2.shape[1] == data_1.kf[ind_min].shape[0]


    # change path for saving figures
    os.chdir(output_folder)
    print(">> ", os.getcwd())

    # Data (M1, M2, H)
    dgmgp = SingleBindGMGP(
        X_train=[data_1.X_train_norm[0], data_2.X_train_norm[0], data_1.X_train_norm[1]],
        Y_train=[data_1.Y_train_norm[0][:, ::n_save], Y_lf_norm_2[:, ::n_save], data_1.Y_train_norm[1][:, ::n_save]],
        n_fidelities=n_fidelities,
        n_samples=500,
        optimization_restarts=n_optimization_restarts,
        ARD_last_fidelity=ARD_last_fidelity,
    )

    # Single-fidelity
    # high-fidelity only emulator
    hf_only = SingleBinGP(data_1.X_train_norm[-1], data_1.Y_train[-1][:, ::n_save])
    lf_only_1 = SingleBinGP(data_1.X_train_norm[0], data_1.Y_train[0][:, ::n_save])
    lf_only_2 = SingleBinGP(data_2.X_train_norm[0], Y_lf_2[:, ::n_save])

    # optimize each model
    hf_only.optimize_restarts(n_optimization_restarts=n_optimization_restarts)
    lf_only_1.optimize_restarts(n_optimization_restarts=n_optimization_restarts)
    lf_only_2.optimize_restarts(n_optimization_restarts=n_optimization_restarts)

    # testing set
    # import pdb
    # pdb.set_trace()

    means_dgmgp, vars_dgmgp, pred_exacts_dgmgp = validate_mf(data_1, model=dgmgp)
    means_hfonly, vars_hfonly, pred_exacts_hfonly = validate_sf(data_1, model=hf_only)
    means_lfonly_1, vars_lfonly_1, pred_exacts_lfonly_1 = validate_sf(data_1, model=lf_only_1)
    means_lfonly_2, vars_lfonly_2, pred_exacts_lfonly_2 = validate_sf(data_1, model=lf_only_2)

    # versus HF
    do_emulator_error_plots(
        data_1,
        means_dgmgp,
        means_hfonly,
        pred_exacts_dgmgp,
        pred_exacts_hfonly,
        label_mf="dGMGP",
        label_sf="HF only",
        figure_name="dGMGP",
    )
    do_emulator_error_plots(
        data_1,
        means_dgmgp,
        means_lfonly_1,
        pred_exacts_dgmgp,
        pred_exacts_lfonly_1,
        label_mf="dGMGP",
        label_sf="LF only (1)",
        figure_name="dGMGP_lf1",
    )
    # versus LF
    do_emulator_error_plots(
        data_1,
        means_dgmgp,
        means_lfonly_2,
        pred_exacts_dgmgp,
        pred_exacts_lfonly_2,
        label_mf="dGMGP",
        label_sf="LF only (2)",
        figure_name="dGMGP_lf2",
    )

    # pred/exact plot
    do_pred_exact(data_1, means_dgmgp, pred_exacts_dgmgp, label_mf="dGMGP", figure_name="dGMGP")

    # # saving hyperparameters
    # with open("ar1.json", "w") as f:
    #     json.dump(ar1.to_dict(), f, indent=2)

    # with open("nargp.json", "w") as f:
    #     json.dump(nargp.to_dict(), f, indent=2)

    # with open("hf_only.json", "w") as f:
    #     json.dump(hf_only.to_dict(), f, indent=2)

    # with open("lf_only.json", "w") as f:
    #     json.dump(lf_only.to_dict(), f, indent=2)

    with open("dgmgp.json", "w") as f:
        json.dump(dgmgp.to_dict(), f, indent=2)


    # saving AR1
    os.makedirs("dGMGP/", exist_ok=True)

    np.savetxt(os.path.join("dGMGP", "all_gp_mean"), np.array(means_dgmgp))
    np.savetxt(os.path.join("dGMGP", "all_gp_var"), np.array(vars_dgmgp))
    np.savetxt(os.path.join("dGMGP", "pred_exacts"), np.array(pred_exacts_dgmgp))
    np.savetxt(os.path.join("dGMGP", "all_true"), np.array(data_1.Y_test[0]))
    np.savetxt(os.path.join("dGMGP", "kf"), np.array(data_1.kf))
    # [HF] also save the predictions from hf-only
    np.savetxt(os.path.join("dGMGP", "all_hf_gp_mean"), np.array(means_hfonly))
    np.savetxt(os.path.join("dGMGP", "all_hf_gp_var"), np.array(vars_hfonly))
    np.savetxt(os.path.join("dGMGP", "pred_exacts_hf"), np.array(pred_exacts_hfonly))
    # [LF] also save the predictions from lf-only
    np.savetxt(os.path.join("dGMGP", "all_lf_gp_mean_1"), np.array(means_lfonly_1))
    np.savetxt(os.path.join("dGMGP", "all_lf_gp_var_1"), np.array(vars_lfonly_1))
    np.savetxt(os.path.join("dGMGP", "pred_exacts_lf_1"), np.array(pred_exacts_lfonly_1))

    np.savetxt(os.path.join("dGMGP", "all_lf_gp_mean_2"), np.array(means_lfonly_2))
    np.savetxt(os.path.join("dGMGP", "all_lf_gp_var_2"), np.array(vars_lfonly_2))
    np.savetxt(os.path.join("dGMGP", "pred_exacts_lf_2"), np.array(pred_exacts_lfonly_2))

    # back to root folder
    os.chdir(old_dir)

    return data_1.kf[::n_save], pred_exacts_dgmgp

def validate_mf(data: PowerSpecs, model: SingleBinNonLinearGP, fidelity: int = 1):
    """
    Validate the trained MFEmulators
    """
    x_test, y_test = data.X_test_norm[0], data.Y_test[0][:, ::n_save]

    mean, var = model.predict(x_test)

    # dx = x dlog(x) = log(10) x dlog10(x)
    # dlog10(x) = d(log(x) / log(10))
    vars = (10 ** mean * np.log(10) * np.sqrt(var))**2

    # predicted/exact
    pred_exacts = 10 ** mean / 10 ** y_test

    return mean, var, pred_exacts


def validate_sf(data: PowerSpecs, model: SingleBinGP):
    """
    Validate the trained single-fidelity emulator
    """
    all_means = []
    all_vars = []
    all_pred_exacts = []
    for n_validations, (x_test, y_test) in enumerate(
        zip(data.X_test_norm[0], data.Y_test[0][:, ::n_save])
    ):
        mean, var = model.predict(x_test[None, :])

        all_means.append(10 ** mean[0])
        # dx = x dlog(x) = log(10) x dlog10(x)
        # dlog10(x) = d(log(x) / log(10))
        all_vars.append((10 ** mean[0] * np.log(10) * np.sqrt(var[0]))**2)

        # predicted/exact
        all_pred_exacts.append(10 ** mean[0] / 10 ** y_test)

    return all_means, all_vars, all_pred_exacts

def do_emulator_error_plots(
    data: PowerSpecs,
    means_mf: List[np.ndarray],
    means_sf: List[np.ndarray],
    pred_exacts_mf: List[np.ndarray],
    pred_exacts_sf: List[np.ndarray],
    label_mf: str = "NARGP",
    label_sf: str = "HF only",
    figure_name: str = "",
):
    """
    1. predicted / exact power spectrum
    2. absolute error plot
    """

    # mean emulation error
    emulator_errors = np.abs(np.array(pred_exacts_mf) - 1)
    plt.loglog(
        10 ** data.kf[::n_save], np.mean(emulator_errors, axis=0), label=label_mf, color="C0"
    )
    plt.fill_between(
        10 ** data.kf[::n_save],
        y1=np.min(emulator_errors, axis=0),
        y2=np.max(emulator_errors, axis=0),
        color="C0",
        alpha=0.3,
    )

    emulator_errors = np.abs(np.array(pred_exacts_sf) - 1)
    plt.loglog(
        10 ** data.kf[::n_save], np.mean(emulator_errors, axis=0), label=label_sf, color="C1"
    )
    plt.fill_between(
        10 ** data.kf[::n_save],
        y1=np.min(emulator_errors, axis=0),
        y2=np.max(emulator_errors, axis=0),
        color="C1",
        alpha=0.3,
    )
    plt.legend()
    plt.ylabel(r"$| P_\mathrm{predicted}(k) / P_\mathrm{true}(k) - 1|$")
    plt.xlabel(r"$k (h/\mathrm{Mpc})$")
    save_figure("absolute_errors_" + figure_name)
    plt.close()
    plt.clf()


def do_pred_exact(
    data: PowerSpecs,
    means_mf: List[np.ndarray],
    pred_exacts_mf: List[np.ndarray],
    label_mf: str = "NARGP",
    figure_name: str = "",
):
    """
    Pred/Exact plot
    """
    for i, pred_exact_mf in enumerate(pred_exacts_mf):
        if i == 0:
            plt.semilogx(
                10 ** data.kf[::n_save], pred_exact_mf, label=label_mf, color="C{}".format(i)
            )
        else:
            plt.semilogx(10 ** data.kf[::n_save], pred_exact_mf, color="C{}".format(i))

    plt.legend()
    plt.ylim(0.96, 1.06)
    plt.xlabel(r"$k (h/\mathrm{Mpc})$")
    plt.ylabel(r"$\mathrm{Predicted/Exact}$")
    save_figure("predict_exact_" + figure_name)
    plt.close()
    plt.clf()
