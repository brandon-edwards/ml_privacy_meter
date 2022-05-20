#!usr/bin/env python
# -*- coding: utf-8 -*-

import os, shutil, argparse
from datetime import date
from pathlib import Path


# main function
if __name__ == "__main__":
    copyrightMessage = (
        "Contact: software@cbica.upenn.edu\n\n"
        + "This program is NOT FDA/CE approved and NOT intended for clinical use.\nCopyright (c) "
        + str(date.today().year)
        + " University of Pennsylvania. All rights reserved."
    )

    cwd = Path(__file__).resolve().parent
    all_files_and_folders = os.listdir(cwd)

    parser = argparse.ArgumentParser(
        prog="GANDLF_Experiment_Submitter",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Submit GaNDLF experiments on CUBIC Cluster.\n\n"
        + copyrightMessage,
    )

    parser.add_argument(
        "-r",
        "--runnerscript",
        metavar="",
        default=os.path.join(cwd, "runner.sh"),
        type=str,
        help="'runner.sh' script to be called.",
    )
    parser.add_argument(
        "-e",
        "--email",
        metavar="",
        default="patis@upenn.edu",
        type=str,
        help="Email address to be used for notifications.",
    )
    parser.add_argument(
        "-gpu",
        "--gputype",
        metavar="",
        default="gpu",
        type=str,
        help="The parameter to pass after '-l' to the submit command.",
    )

    args = parser.parse_args()

    # configs that SP has trained for exp ID 20220428_2024
    noise_multipliers = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10]
    max_grad_norms = [1.0, 2.0, 5.0, 10]

    ## 20220514: each attack takes ~40 min on P100 node on CUBIC-cluster

    base_model_path = "/cbica/home/patis/comp_space/testing/gandlf_dp_experiments_20220428_2024"
    
    for noise_multiplier in noise_multipliers:
        # ".2f" is needed because that's how strings are getting saved
        nm_str = format(noise_multiplier, ".2f")
        nm_dir = os.path.join(base_model_path, "nm_" + nm_str)
        for grad in max_grad_norms:
            gm_str = format(grad, ".2f")
            nm_gm_dir = os.path.join(nm_dir, "gm_" + gm_str)
            model_filepath = os.path.join(nm_gm_dir, "imagenet_vgg16_best.pth.tar")

            # GaNDLF config path here
            current_config = os.path.join(nm_dir, "gm_" + gm_str + ".yaml")

            exp_name = nm_str + "_" + gm_str

            command = (
                "qsub -N L_"
                + exp_name
                + " -M "
                + args.email
                + " -l "
                + args.gputype
                + " "
                + args.runnerscript
                + " "
                + exp_name
                + " "
                + current_config
                + " "
                + model_filepath
            )
            print(command)
            os.system(command)

    # baseline model
    nm_dir = os.path.join(base_model_path, "baseline")
    nm_gm_dir = os.path.join(nm_dir, "base")
    current_config = os.path.join(nm_dir, "base.yaml")
    model_filepath = os.path.join(nm_gm_dir, "imagenet_vgg16_best.pth.tar")
    exp_name = "benign"
    
    command = (
        "qsub -N L_"
        + exp_name
        + " -M "
        + args.email
        + " -l "
        + args.gputype
        + " "
        + args.runnerscript
        + " "
        + exp_name
        + " "
        + current_config
        + " "
        + model_filepath
    )
    print(command)
    os.system(command)
