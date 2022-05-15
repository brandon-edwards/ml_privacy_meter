import os, argparse, ast, sys
from pathlib import Path

from functools import partial
import numpy as np
import logging

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torchio import DATA


import tensorflow as tf
# from torch.nn import CrossEntropyLoss

from GANDLF.models import vgg16, imagenet_vgg16 
from GANDLF.parseConfig import parseConfig

from GANDLF.data.loader_restrictor import LoaderRestrictor

import ml_privacy_meter

from GANDLF.utils import populate_header_in_parameters, parseTrainingCSV, populate_channel_keys_in_params
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame 

logger = logging.getLogger()

# Set attack hyperparameters
# This determines how many points per class are used to profile the population loss values
num_data_in_class = 1000

# device = 'cuda'
# exp_name = 'tutorial_pytorch_sbu'

# population_csv_path = "/cbica/home/patis/comp_space/testing/ml_privacy_meter/sbu_new_csv/SBU_pm_population_class_balanced.csv"
# train_csv_path = "/cbica/home/patis/comp_space/testing/ml_privacy_meter/sbu_new_csv/SBU_pm_train_class_balanced.csv"
# test_csv_path =  "/cbica/home/patis/comp_space/testing/ml_privacy_meter/sbu_new_csv/SBU_pm_test_class_balanced.csv"

batch_size = 1
# We will keep this batch size as some code expects it
assert batch_size == 1

# base_model_path = "/cbica/home/patis/comp_space/testing/gandlf_dp_experiments_20220428_2024"
# model_filepath = '/home/aspaul/GaNDLF/experiment_e15_imagenetvgg16_modeleveryepoch/model_dir/imagenet_vgg16_best.pth.tar'

# defining dict for models - key is the string and the value is the transform object
global_models_dict = {
    "vgg16": vgg16,
    "imagenet_vgg16": imagenet_vgg16
}

def gandlf_dict_to_feature(subject_dict, gandlf_config):
    return (torch.cat([subject_dict[key][DATA] for key in gandlf_config["channel_keys"]], 
                             dim=1).float().to(gandlf_config["device"]))
    
def gandlf_dict_to_label(subject_dict, gandlf_config):
    if len(subject_dict["value_0"].detach().cpu().numpy()) != 1:
        raise ValueError("Code expects batch size of one!")
    num_labels = len(gandlf_config['model']['class_list'])
    int_label = int(subject_dict["value_0"].detach().cpu().numpy().item())
    return np.eye(num_labels)[int_label]


# The following function was copied and modified from create_pytorch_objects 
# at https://github.com/sarthakpati/GaNDLF/blob/openfl-integration/GANDLF/compute/generic.py: 

def get_loaders(parameters, train_csv_path, test_csv_path, population_csv_path):
    """
    This function creates the data loaders for the population, train, and test data. The train and 
    test data should be the same size, and the population data should represent data
    from the same distribution as the train and test but not contain a significant 
    amount of training data.
    Args:
        parameters (dict): The parameters dictionary.
        train_csv_path (str): The path to the population CSV file.
        test_csv_path (str): The path to the training CSV file.
        population_csv_path (str): The path to the validation CSV file.
    Returns:
        train_loader (torch.utils.data.DataLoader): The training data loader.
        test_loader (torch.utils.data.DataLoader): The testing data loader.
        population_loader (torch.utils.data.DataLoader): The population data loader.
    """

    # initialize loaders
    train_loader, test_loader, population_loader = None, None, None

    
    # populate the data frames for the train loader (train is False since no augmentations and or patching are wanted)
    train_paths, headers_train = parseTrainingCSV(
        train_csv_path, train=True
    )
    parameters = populate_header_in_parameters(parameters, headers_train)

    training_data_for_torch = ImagesFromDataFrame(
        train_paths, parameters, train=False, loader_type="train"
    )

    # Fetch the appropriate channel keys
    # Getting the channels for training and removing all the non numeric entries from the channels
    parameters = populate_channel_keys_in_params(
        training_data_for_torch, parameters
    )

    # get the train loader
    train_loader = DataLoader(
        training_data_for_torch,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )

    # populate the data frames for the test loader
    test_paths, _ = parseTrainingCSV(
        test_csv_path, train=True
    )    
    test_data_for_torch = ImagesFromDataFrame(
        test_paths, parameters, train=False, loader_type="train"
    )
    # get the test loader
    test_loader = DataLoader(
        test_data_for_torch,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )
    
    # populate the data frames for the population loader (again train is False due to no augmentations wanted)
    population_paths, _ = parseTrainingCSV(
        population_csv_path, train=True
    )
    
    population_data_for_torch = ImagesFromDataFrame(
        population_paths, parameters, train=False, loader_type="train"
    )

    # get the population loader
    population_loader = DataLoader(
        population_data_for_torch,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,  # params["pin_memory_dataloader"], # this is going OOM if True - needs investigation
    )

    # these keys contain generators, and are not needed beyond this point in params
    generator_keys_to_remove = ["optimizer_object", "model_parameters"]
    for key in generator_keys_to_remove:
        parameters.pop(key, None)

    return train_loader, test_loader, population_loader



def get_model_class(parameters):
    """
    This function gets the model class being used from the global models dict.
    Args:
        parameters (dict): The parameters dictionary.
    Returns:
        model_class (torch.nn.Module): The model to use for training.
    """

    # get the model class (here we use a vgg only global models dict since not using this script much, as it will 
    # be replaced when PM code is made more modular)
    model_class = global_models_dict[parameters["model"]["architecture"]]

    return model_class    



if __name__ == '__main__':
    copyrightMessage = (
        "Contact: gandlf@cbica.upenn.edu\n\n"
        + "This program is NOT FDA/CE approved and NOT intended for clinical use.\n"
    )
    parser = argparse.ArgumentParser(
        prog="PrivacyMeter",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Model leakage quantification.\n\n"
        + copyrightMessage,
    )
    parser.add_argument(
        "-cp",
        "--csvpop",
        metavar="",
        type=str,
        help="The CSV to population data",
    )
    parser.add_argument(
        "-ctr",
        "--csvtrain",
        metavar="",
        type=str,
        help="The CSV to train data",
    )
    parser.add_argument(
        "-cts",
        "--csvtest",
        metavar="",
        type=str,
        help="The CSV to test data",
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="",
        type=str,
        help="The configuration file (contains all the information related to the training/inference session)",
    )
    parser.add_argument(
        "-m",
        "--model",
        metavar="",
        type=str,
        help="The saved model file",
    )
    parser.add_argument(
        "-e",
        "--expname",
        metavar="",
        default="tutorial_pytorch_sbu",
        type=str,
        help="The experiment name (default: tutorial_pytorch_sbu)",
    )
    parser.add_argument(
        "-d",
        "--device",
        metavar="",
        default="cuda",
        type=str,
        help="The device to run compute on (default: cuda)",
    )

    args = parser.parse_args()

    # GaNDLF config path here
    gandlf_config_path = os.path.join(args.config)
    gandlf_config = parseConfig(gandlf_config_path)
    gandlf_config['device'] = gandlf_config.get('device', args.device)

    # get dataset (loaders) (preprocess script is what changes here for new dataset)
    target_model_class = get_model_class(parameters=gandlf_config)
    
    train_loader, test_loader, population_loader = get_loaders(parameters=gandlf_config, 
                                                            population_csv_path=args.csvpop, 
                                                            train_csv_path=args.csvtrain, 
                                                            test_csv_path=args.csvtest) 

    fhandler = logging.FileHandler(filename='population_attack_' + args.expname + '.log', mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%Y%m%d')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.CRITICAL)
    logger.critical(f"Running population attack against model {target_model_class}\n")
    logger.critical(f"Using GANDLF config at: {gandlf_config_path}\n")
    logger.critical(f"PM Population data from: {args.csvpop}\n")
    logger.critical(f"PM Train samples from: {args.csvtrain}\n")
    logger.critical(f"PM test samples from: {args.csvtest}\n")            
    logger.critical(f"Model params from file at {args.model}\n")
    
    # original code
    if os.path.isfile(args.model):
        logger.critical(f"Model already trained. Continuing...")
    else:
        raise RuntimeError('This script was not intended to be used if the model is not trained yet.')

    x_train = LoaderRestrictor(base_loader=train_loader, 
                            restrictions= ('feature', None), 
                            dict_to_feature=partial(gandlf_dict_to_feature, gandlf_config=gandlf_config), 
                            dict_to_label=None)
    y_train = LoaderRestrictor(base_loader=train_loader, 
                            restrictions= ('label', None), 
                            dict_to_feature=None, 
                            dict_to_label=partial(gandlf_dict_to_label, gandlf_config=gandlf_config))

    x_test = LoaderRestrictor(base_loader=test_loader, 
                            restrictions= ('feature', None), 
                            dict_to_feature=partial(gandlf_dict_to_feature, gandlf_config=gandlf_config), 
                            dict_to_label=None)
    y_test = LoaderRestrictor(base_loader=test_loader, 
                            restrictions= ('label', None), 
                            dict_to_feature=None, 
                            dict_to_label=partial(gandlf_dict_to_label, gandlf_config=gandlf_config))

    x_population = LoaderRestrictor(base_loader=population_loader, 
                                    restrictions= ('feature', None), 
                                    dict_to_feature=partial(gandlf_dict_to_feature, gandlf_config=gandlf_config), 
                                    dict_to_label=None)
    y_population = LoaderRestrictor(base_loader=population_loader, 
                                    restrictions= ('label', None), 
                                    dict_to_feature=None, 
                                    dict_to_label=partial(gandlf_dict_to_label, gandlf_config=gandlf_config))
    

    # create population attack object
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    # loss_fn = CrossEntropyLoss(reduction='none')
    population_attack_obj = ml_privacy_meter.attack.population_meminf.PopulationAttack(
        exp_name=args.expname,
        gandlf_config=gandlf_config,
        x_population=x_population, y_population=y_population,
        x_target_train=x_train, y_target_train=y_train,
        x_target_test=x_test, y_target_test=y_test,
        target_model_filepath=args.model,
        target_model_type=ml_privacy_meter.utils.attack_utils.MODEL_TYPE_PYTORCH,
        target_model_class=target_model_class,  # pass in the model class for pytorch
        loss_fn=loss_fn,
        num_data_in_class=num_data_in_class, 
        num_classes=len(gandlf_config['model']['class_list']), 
        seed=1234, 
        device=args.device, 
        logger=logger
    )

    population_attack_obj.prepare_attack()

    alphas = [0.1, 0.3, 0.5]

    logger.critical(f"Running with alphas: {alphas}")

    population_attack_obj.run_attack(alphas=alphas)

    population_attack_obj.visualize_attack(alphas=alphas)
