import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


# TODO: implement this based on type of model: tensorflow, openVINO, etc.
def get_predictions(model, data):
    return data


def get_per_class_indices(x, y, num_data_in_class, seed):
    num_classes = y.shape[1]

    per_class_splitter = StratifiedShuffleSplit(n_splits=1,
                                                train_size=(num_data_in_class * num_classes),
                                                test_size=100,
                                                random_state=seed)

    split_indices = []
    for indices, _ in per_class_splitter.split(x, y):
        split_indices = indices

    per_class_indices = []
    for c in range(num_classes):
        indices = []
        for idx in split_indices:
            x_point, y_point = x[idx], y[idx]
            if c == np.argmax(y_point):
                indices.append(idx)
        # print(f"Number of samples from class {c} = {len(indices)}")
        per_class_indices.append(indices)

    return per_class_indices


def calculate_loss_threshold(alpha, loss_distribution):
    threshold = np.quantile(loss_distribution, q=alpha, interpolation='lower')
    return threshold


class PopulationAttack:
    def __init__(self, exp_name, x_population, y_population,
                 x_target_train, y_target_train,
                 x_target_test, y_target_test,
                 target_model, loss_fn,
                 num_data_in_class, seed):
        self.x_population = x_population
        self.y_population = y_population
        self.x_target_train = x_target_train
        self.y_target_train = y_target_train
        self.x_target_test = x_target_test
        self.y_target_test = y_target_test
        self.target_model = target_model
        self.loss_fn = loss_fn
        self.num_data_in_class = num_data_in_class
        self.seed = seed

        self.num_classes = self.y_population.shape[0]

        # create results directory
        self.attack_results_dirpath = f'logs/population_attack_{exp_name}/'
        if not os.path.isdir(Path(self.attack_results_dirpath)):
            os.mkdir(Path(self.attack_results_dirpath))

    def prepare_attack(self):
        """
        Compute and save loss values of the target model on its train and test data.
        """
        print("Computing and saving train and test losses of the target model...")

        train_losses = self.loss_fn(
            y_true=self.y_target_train,
            y_pred=get_predictions(self.target_model, self.x_target_train)
        )
        test_losses = self.loss_fn(
            y_true=self.y_target_test,
            y_pred=get_predictions(self.target_model, self.x_target_test)
        )

        np.savez(f"{self.attack_results_dirpath}/target_model_losses",
                 train_losses=train_losses,
                 test_losses=test_losses)

    def run_attack(self, alphas):
        """
        Run the population attack on the target model.
        """
        print("Running the population attack on the target model...")

        # get train and test losses
        losses_filepath = f"{self.attack_results_dirpath}/target_model_losses.npz"
        if os.path.isfile(losses_filepath):
            with np.load(losses_filepath, allow_pickle=True) as data:
                train_losses = data['train_losses'][()]
                test_losses = data['test_losses'][()]

        # get per-class indices
        per_class_indices = get_per_class_indices(
            x=self.x_population, y=self.y_population,
            num_data_in_class=self.num_data_in_class,
            seed=self.seed)

        # load per class losses, compute them if they don't exist
        filepath = f"{self.attack_results_dirpath}/target_model_pop_losses_{self.num_data_in_class}.npz"
        if os.path.isfile(filepath):
            with np.load(filepath, allow_pickle=True) as data:
                pop_losses = data['pop_losses'][()]
        else:
            pop_losses = []
            for c in range(self.num_classes):
                indices = per_class_indices[c]
                x_class, y_class = self.x_population[indices], self.y_population[indices]
                losses = self.loss_fn(
                    y_true=y_class,
                    y_pred=get_predictions(self.target_model, x_class)
                )
                pop_losses.append(losses)
            np.savez(f"{self.attack_results_dirpath}/target_model_pop_losses_{self.num_data_in_class}",
                     pop_losses=pop_losses)

        # run the attack for every alpha
        for alpha in alphas:
            print(f"For alpha = {alpha}...")
            per_class_thresholds = []
            for c in range(self.num_classes):
                threshold = calculate_loss_threshold(alpha, pop_losses[c])
                per_class_thresholds.append(threshold)

            # generate predictions: <= threshold, output '1' (member) else '0' (non-member)
            preds = []
            for (loss, label) in zip(train_losses, self.y_target_train):
                c = int(np.argmax(label))
                threshold = per_class_thresholds[c]
                if loss <= threshold:
                    preds.append(1)
                else:
                    preds.append(0)

            for (loss, label) in zip(test_losses, self.y_target_test):
                c = int(np.argmax(label))
                threshold = per_class_thresholds[c]
                if loss <= threshold:
                    preds.append(1)
                else:
                    preds.append(0)

            y_eval = [1] * len(train_losses)
            y_eval.extend([0] * len(test_losses))

            # save attack results
            acc = accuracy_score(y_eval, preds)
            roc_auc = roc_auc_score(y_eval, preds)
            tn, fp, fn, tp = confusion_matrix(y_eval, preds).ravel()
            np.savez(f"{self.attack_results_dirpath}/attack_results_{alpha}_{self.num_data_in_class}",
                     true_labels=y_eval, preds=preds,
                     alpha=alpha, num_data_in_class=self.num_data_in_class,
                     per_class_thresholds=per_class_thresholds,
                     acc=acc, roc_auc=roc_auc,
                     tn=tn, fp=fp, tp=tp, fn=fn)

            print(
                f"Population attack performance:\n"
                f"Number of points in class: {self.num_data_in_class}\n"
                f"Accuracy = {acc}\n"
                f"ROC AUC Score = {roc_auc}\n"
                f"FPR: {fp / (fp + tn)}\n"
                f"TN, FP, FN, TP = {tn, fp, fn, tp}"
            )
