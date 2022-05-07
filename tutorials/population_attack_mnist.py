import os
from pathlib import Path

from functools import partial
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torchio import DATA

import tensorflow as tf

from GANDLF.models import vgg11, vgg13, vgg16, vgg19, imagenet_vgg16 
from GANDLF.parseConfig import parseConfig

from GANDLF.data.loader_restrictor import LoaderRestrictor

import ml_privacy_meter

from GANDLF.utils import populate_header_in_parameters, parseTrainingCSV, populate_channel_keys_in_params
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame 


# Set attack hyperparameters
# This determines how many points per class are used to profile the population loss values
num_data_in_class = 1000

device = 'cuda'

# GaNDLF config path here
gandlf_config_path = '/home/edwardsb/projects/SBU-TIL/configs/brandon_quick_test_config_mnist.yaml'

gandlf_config = parseConfig(gandlf_config_path)
gandlf_config['device'] = gandlf_config.get('device', device)

population_csv_path = "/home/edwardsb/projects/SBU-TIL/MNIST_Data/MNIST_pm_population_small.csv"
train_csv_path = "/home/edwardsb/projects/SBU-TIL/MNIST_Data/MNIST_pm_train_small.csv"
test_csv_path =  "/home/edwardsb/projects/SBU-TIL/MNIST_Data/MNIST_pm_test_small.csv"

batch_size = 1
# We will keep this batch size as some code expects it
assert batch_size == 1

model_name = 'tutorial_pytorch_mnist'
exp_name = 'tutorial_pytorch_mnist'

model_filepath = '/raid/edwardsb/models/projects/SBU-TIL/MNIST/imagenet_vgg16_best.pth.tar'
    

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

def get_model_class_and_loaders(parameters, train_csv_path, test_csv_path, population_csv_path):
    """
    This function gets the model class being used from the global models dict and 
    creates the data loaders for the population, train, and test data. The train and 
    test data should be the same size, and the population data should represent data
    from the same distribution as the train and test but not contain a significant 
    amount of training data.
    Args:
        parameters (dict): The parameters dictionary.
        train_csv_path (str): The path to the population CSV file.
        test_csv_path (str): The path to the training CSV file.
        population_csv_path (str): The path to the validation CSV file.
    Returns:
        model_class (torch.nn.Module): The model to use for training.
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

    # get the model class (here we use a vgg only global models dict since not using this script much, as it will 
    # be replaced when PM code is made more modular)
    model_class = global_models_dict[parameters["model"]["architecture"]]

    # these keys contain generators, and are not needed beyond this point in params
    generator_keys_to_remove = ["optimizer_object", "model_parameters"]
    for key in generator_keys_to_remove:
        parameters.pop(key, None)

    return model_class, train_loader, test_loader, population_loader



if __name__ == '__main__':

    # get dataset (loaders) (preprocess script is what changes here for new dataset)
    target_model_class, train_loader, test_loader, population_loader = get_model_class_and_loaders(parameters=gandlf_config, 
                                                                                                   population_csv_path=population_csv_path, 
                                                                                                   train_csv_path=train_csv_path, 
                                                                                                   test_csv_path=test_csv_path)
  
    if os.path.isfile(model_filepath):
        print(f"Model already trained. Continuing...")
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
    population_attack_obj = ml_privacy_meter.attack.population_meminf.PopulationAttack(
        exp_name=exp_name,
        gandlf_config=gandlf_config,
        x_population=x_population, y_population=y_population,
        x_target_train=x_train, y_target_train=y_train,
        x_target_test=x_test, y_target_test=y_test,
        target_model_filepath=model_filepath,
        target_model_type=ml_privacy_meter.utils.attack_utils.MODEL_TYPE_PYTORCH,
        target_model_class=target_model_class,  # pass in the model class for pytorch
        loss_fn=loss_fn,
        num_data_in_class=num_data_in_class, 
        num_classes=len(gandlf_config['model']['class_list']), 
        seed=1234, 
        device=device
    )

    population_attack_obj.prepare_attack()

    alphas = [0.1, 0.3, 0.5]
    population_attack_obj.run_attack(alphas=alphas)

    population_attack_obj.visualize_attack(alphas=alphas)
