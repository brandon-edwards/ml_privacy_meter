import os
import time

import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.compat.v1.train import Saver

# from openvino.inference_engine import IECore
import torch

from GANDLF.utils import load_model

MODEL_TYPE_OPENVINO = 'openvino'
MODEL_TYPE_TENSORFLOW = 'tensorflow'
MODEL_TYPE_PYTORCH = 'pytorch'

def sanity_check(layers, layers_to_exploit):
    """
    Basic sanity check for layers and gradients to exploit based on model layers
    """
    if layers_to_exploit and len(layers_to_exploit):
        assert np.max(layers_to_exploit) <= len(layers),\
            "layer index greater than the last layer"


def time_taken(self, start_time, end_time):
    """
    Calculates difference between 2 times
    """
    delta = end_time - start_time
    hours = int(delta / 3600)
    delta -= hours * 3600
    minutes = int(delta / 60)
    delta -= minutes * 60
    seconds = delta
    return hours, minutes, np.int(seconds)


def get_predictions(model_filepath, model_type, data, gandlf_config, model_class=None, device='cpu'):
    if model_type == MODEL_TYPE_OPENVINO:
        raise ValueError("OpenVINO models are not supported.")
    elif model_type == MODEL_TYPE_TENSORFLOW:
        model = tf.keras.models.load_model(model_filepath)
        predictions = model(data)
    elif model_type == MODEL_TYPE_PYTORCH:
        model = model_class(parameters=gandlf_config)  # pytorch models need to be instantiated
        main_dict = load_model(model_filepath, device)
        model.load_state_dict(main_dict["model_state_dict"])

        # model.load_state_dict(torch.load(model_filepath))
        model.to(device)
        model.eval()

        predictions = np.array([])
        for idx, batch in enumerate(data):
            # TODO: This code should really live in the data loader
            batch = torch.squeeze(batch, dim=-1)
            #batch = torch.transpose(batch, [0,])

            batch_predictions = model(batch).detach().cpu().numpy()
            if idx == 0:
                predictions = batch_predictions
            else:
                predictions = np.append(predictions, batch_predictions, 0)
    else:
        raise ValueError("Please specify one of the supported model types: `openvino`, `tensorflow`, or `pytorch`!")
    return predictions

def get_labels_from_batch(batch_loader_labels, num_classes, labels_out=None):
    # Produces one-hot encoded labels for batch, when the expected
    # batch_labels coming in are an iterator of integer labels
    if batch_loader_labels.shape == (num_classes,):
        if labels_out is None:
            return np.expand_dims(batch_loader_labels, 0)
        else:
            return np.append(labels_out, np.expand_dims(batch_loader_labels, 0), 0)
    else:
    # currently expecting a single one-hot encoded class
        raise ValueError(f"Need to change code to account for larger batch than size 1 since this batch first dimension is: {len(batch_loader_labels)}")
    

def get_labels(label_loader_restrictor, num_classes):
    for idx, batch_loader_labels in enumerate(label_loader_restrictor):
        if idx == 0:
            labels_out = get_labels_from_batch(batch_loader_labels=batch_loader_labels, 
                                               num_classes=num_classes, 
                                               labels_out=None)
        else:
            labels_out = get_labels_from_batch(batch_loader_labels=batch_loader_labels, 
                                               num_classes=num_classes, 
                                               labels_out=labels_out)
    return labels_out

def get_per_class_indices(y, num_data_in_class, seed):
    num_classes = y.shape[1]
    # TODO: Figure out correct setting for train and test size below
    # TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    per_class_splitter = StratifiedShuffleSplit(n_splits=1,
                                                train_size=(num_data_in_class * num_classes)/len(y),
                                                test_size=100,
                                                random_state=seed)
    """
    per_class_splitter = StratifiedShuffleSplit(n_splits=1,
                                                train_size=0.5,
                                                random_state=seed)
    split_indices = []
    # previously the first y was given as x, but we do not need it and x is a loader_restrictor so don't want to use
    for indices, _ in per_class_splitter.split(y, y):
        # this iterator is length one since n_splits above is 1
        split_indices = indices

    per_class_indices = []
    for c in range(num_classes):
        indices = []
        for idx in split_indices:
            y_point = y[idx]
            if c == np.argmax(y_point):
                indices.append(idx)
        # print(f"Number of samples from class {c} = {len(indices)}")
        per_class_indices.append(indices)

    return per_class_indices


def calculate_loss_threshold(alpha, loss_distribution):
    threshold = np.quantile(loss_distribution, q=alpha, interpolation='lower')
    return threshold


class attack_utils():
    """
    Utilities required for conducting membership inference attack
    """

    def __init__(self, directory_name='latest'):
        self.root_dir = os.path.abspath(os.path.join(
                                        os.path.dirname(__file__),
                                        "..", ".."))
        self.log_dir = os.path.join(self.root_dir, "logs")
        self.aprefix = os.path.join(self.log_dir,
                                    directory_name,
                                    "attack",
                                    "model_checkpoints")
        self.dataset_directory = os.path.join(self.root_dir, "datasets")

        if not os.path.exists(self.aprefix):
            os.makedirs(self.aprefix)
        if not os.path.exists(self.dataset_directory):
            os.makedirs(self.dataset_directory)

    def get_gradshape(self, variables, layerindex):
        """
        Returns the shape of gradient matrices
        Args:
        -----
        model: model to attack 
        """
        g = (layerindex-1)*2
        gradshape = variables[g].shape
        return gradshape

    def get_gradient_norm(self, gradients):
        """
        Returns the norm of the gradients of loss value
        with respect to the parameters

        Args:
        -----
        gradients: Array of gradients of a batch 
        """
        gradient_norms = []
        for gradient in gradients:
            summed_squares = [K.sum(K.square(g)) for g in gradient]
            norm = K.sqrt(sum(summed_squares))
            gradient_norms.append(norm)
        return gradient_norms

    def get_entropy(self, model, features, output_classes):
        """
        Calculates the prediction uncertainty
        """
        entropyarr = []
        for feature in features:
            feature = tf.reshape(feature, (1, len(feature.numpy())))
            predictions = model(feature)
            predictions = tf.nn.softmax(predictions)
            mterm = tf.reduce_sum(input_tensor=tf.multiply(predictions,
                                                           np.log(predictions)))
            entropy = (-1/np.log(output_classes)) * mterm
            entropyarr.append(entropy)
        return entropyarr

    def split(self, x):
        """
        Splits the array into number of elements equal to the
        size of the array. This is required for per example
        computation.
        """
        split_x = tf.split(x, len(x.numpy()))
        return split_x

    def get_savers(self, attackmodel):
        """
        Creates prefixes for storing classification and inference
        model
        """
        # Prefix for storing attack model checkpoints
        prefix = os.path.join(self.aprefix, "ckpt")
        # Saver for storing checkpoints
        attacksaver = Saver(attackmodel.variables)
        return prefix, attacksaver

    def createOHE(self, num_output_classes):
        """
        creates one hot encoding matrix of all the vectors
        in a given range of 0 to number of output classes.
        """
        return tf.one_hot(tf.range(0, num_output_classes),
                          num_output_classes,
                          dtype=tf.float32)

    def one_hot_encoding(self, labels, ohencoding):
        """
        Creates a one hot encoding of the labels used for 
        inference model's sub neural network

        Args: 
        ------
        zero_index: `True` implies labels start from 0
        """
        labels = tf.cast(labels, tf.int64).numpy()
        return tf.stack(list(map(lambda x: ohencoding[x], labels)))

    def intersection(self, to_remove, remove_from, batch_size):
        """
        Finds the intersection between `to_remove` and `remove_from`
        and removes this intersection from `remove_from` 
        """
        to_remove = to_remove.unbatch()
        remove_from = remove_from.unbatch()

        m1, m2 = dict(), dict()
        for example in to_remove:
            hashval = hash(bytes(np.array(example)))
            m1[hashval] = example
        for example in remove_from:
            hashval = hash(bytes(np.array(example)))
            m2[hashval] = example

        # Removing the intersection
        extracted = {key: value for key,
                     value in m2.items() if key not in m1.keys()}
        dataset = extracted.values()
        features, labels = [], []
        for d in dataset:
            features.append(d[0])
            labels.append(d[1])
        finaldataset = tf.compat.v1.data.Dataset.from_tensor_slices(
            (features, labels))
        return finaldataset.batch(batch_size=batch_size)
