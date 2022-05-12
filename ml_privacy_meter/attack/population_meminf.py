import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score

from ml_privacy_meter.utils.attack_utils import get_predictions, get_per_class_indices
from ml_privacy_meter.utils.attack_utils import calculate_loss_threshold, get_labels



class PopulationAttack:
    def __init__(self, exp_name, x_population, y_population,
                 x_target_train, y_target_train,
                 x_target_test, y_target_test,
                 target_model_filepath, target_model_type,
                 loss_fn, num_data_in_class, num_classes, seed, gandlf_config,
                 logger, target_model_class=None, device='cpu'):

        # The data below come in as GANDLF.data.loader_restrictor.LoaderRestrictor
        # objects. Numpy slicing funcionality is replaced by using the 
        # LoaderRestrictor method, 'set_idx_restrictions'. All features are ultimately
        # iterated over inside get_predictions at which point LoaderFestrictor produces tensors. 
        # Predictions coming out of get_predictions is a numpy array. Labels are converted
        # to numpy and one-hot encoded immediately below upon assignment to class attritutes.

        self.x_population = x_population
        self.y_population = get_labels(y_population, num_classes=num_classes)
        self.x_target_train = x_target_train
        self.y_target_train = get_labels(y_target_train, num_classes=num_classes)
        self.x_target_test = x_target_test
        self.y_target_test = get_labels(y_target_test, num_classes=num_classes)
        self.target_model_filepath = target_model_filepath
        self.target_model_type = target_model_type
        self.target_model_class = target_model_class
        self.loss_fn = loss_fn
        self.num_data_in_class = num_data_in_class
        self.seed = seed
        self.gandlf_config=gandlf_config

        self.num_classes = num_classes
        self.device=device
        self.logger = logger

        # create results directory
        self.attack_results_dirpath = f'logs/population_attack_{exp_name}/'
        if not os.path.isdir(Path(self.attack_results_dirpath)):
            os.mkdir(Path(self.attack_results_dirpath))

    def prepare_attack(self):
        """
        Compute and save loss values of the target model on its train and test data.
        """
        self.logger.critical("Computing and saving train and test losses of the target model...")
        train_losses = self.loss_fn(
            y_true=self.y_target_train,
            y_pred=get_predictions(
                model_filepath=self.target_model_filepath,
                model_type=self.target_model_type,
                data=self.x_target_train,
                model_class=self.target_model_class, 
                gandlf_config=self.gandlf_config,
                device=self.device
            )
        )
        test_losses = self.loss_fn(
            y_true=self.y_target_test,
            y_pred=get_predictions(
                model_filepath=self.target_model_filepath,
                model_type=self.target_model_type,
                data=self.x_target_test,
                model_class=self.target_model_class, 
                gandlf_config=self.gandlf_config, 
                device=self.device
            )
        )

        np.savez(f"{self.attack_results_dirpath}/target_model_losses",
                 train_losses=train_losses,
                 test_losses=test_losses)

    def run_attack(self, alphas):
        """
        Run the population attack on the target model.
        """
        self.logger.critical("Running the population attack on the target model...")

        # get train and test losses
        losses_filepath = f"{self.attack_results_dirpath}/target_model_losses.npz"
        if os.path.isfile(losses_filepath):
            with np.load(losses_filepath, allow_pickle=True) as data:
                train_losses = data['train_losses'][()]
                test_losses = data['test_losses'][()]
        else:
            self.prepare_attack()

        # get per-class indices
        per_class_indices = get_per_class_indices(
            y=self.y_population,
            num_data_in_class=self.num_data_in_class,
            seed=self.seed
        )

        # load per class losses, compute them if they don't exist
        filepath = f"{self.attack_results_dirpath}/target_model_pop_losses_{self.num_data_in_class}.npz"
        if os.path.isfile(filepath):
            with np.load(filepath, allow_pickle=True) as data:
                pop_losses = data['pop_losses'][()]
        else:
            pop_losses = []
            for c in range(self.num_classes):
                indices = per_class_indices[c]
                further_restricted_x_population = self.x_population.copy()
                further_restricted_x_population.set_idx_restrictions(indices)
                x_class = further_restricted_x_population  
                y_class = self.y_population[indices]
                losses = self.loss_fn(
                    y_true=y_class,
                    y_pred=get_predictions(
                        model_filepath=self.target_model_filepath,
                        model_type=self.target_model_type,
                        data=x_class,
                        model_class=self.target_model_class, 
                        device=self.device, 
                        gandlf_config=self.gandlf_config
                    )
                )
                pop_losses.append(losses)
            np.savez(f"{self.attack_results_dirpath}/target_model_pop_losses_{self.num_data_in_class}",
                     pop_losses=pop_losses)

        # run the attack for every alpha
        for alpha in alphas:
            self.logger.critical(f"For alpha = {alpha}...\n")
            per_class_thresholds = []
            for c in range(self.num_classes):
                threshold = calculate_loss_threshold(alpha, pop_losses[c])
                per_class_thresholds.append(threshold)

            # generate per class membership predictions: <= threshold, output '1' (member) else '0' (non-member)
            # and record per class membership ground truth(y_eval)
            preds = {c: [] for c in range(self.num_classes)}
            y_eval = {c: [] for c in range(self.num_classes)}
            for (loss, label) in zip(train_losses, self.y_target_train):
                c = int(np.argmax(label))
                threshold = per_class_thresholds[c]
                if loss <= threshold:
                    preds[c].append(1)
                else:
                    preds[c].append(0)
                y_eval[c].append(1)

            for (loss, label) in zip(test_losses, self.y_target_test):
                c = int(np.argmax(label))
                threshold = per_class_thresholds[c]
                if loss <= threshold:
                    preds[c].append(1)
                else:
                    preds[c].append(0)
                y_eval[c].append(0)

            # save attack results
            acc = {c: accuracy_score(y_eval[c], preds[c]) for c in range(self.num_classes)}
            roc_auc = {c: roc_auc_score(y_eval[c], preds[c]) for c in range(self.num_classes)}
            tn, fp, fn, tp = {}, {}, {}, {}
            for c in range(self.num_classes):
                tn[c], fp[c], fn[c], tp[c] = confusion_matrix(y_eval[c], preds[c]).ravel()
                np.savez(f"{self.attack_results_dirpath}/attack_results_alpha_{alpha}_numdatinclass_{self.num_data_in_class}_class_{c}",
                        true_labels=y_eval[c], preds=preds[c],
                        alpha=alpha, num_data_in_class=self.num_data_in_class,
                        per_class_thresholds=per_class_thresholds,
                        acc=acc[c], roc_auc=roc_auc[c],
                        tn=tn[c], fp=fp[c], tp=tp[c], fn=fn[c])
                
                self.logger.critical(
                    f"\nPopulation attack performance:\n"
                    f"Number of points in class: {self.num_data_in_class}\n"
                    f"Accuracy for class {c} = {acc[c]}\n"
                    f"ROC AUC Score for class {c} = {roc_auc[c]}\n"
                    f"FPR for class {c}: {fp[c] / (fp[c] + tn[c])}\n"
                    f"TN, FP, FN, TP for class {c} = {tn[c], fp[c], fn[c], tp[c]}"
                )

    def visualize_attack(self, alphas):
        alphas = sorted(alphas)
        auc_value = {}

        tpr_values = {c: [] for c in range(self.num_classes)}
        fpr_values = {c: [] for c in range(self.num_classes)}

        class_labels_of_preds = np.concatenate([
            np.argmax(self.y_target_train, axis=1),
            np.argmax(self.y_target_test, axis=1)
        ])

        losses_filepath = f"{self.attack_results_dirpath}/target_model_losses.npz"
        if os.path.isfile(losses_filepath):
            with np.load(losses_filepath, allow_pickle=True) as loss_data:
                train_losses = loss_data['train_losses'][()]
                test_losses = loss_data['test_losses'][()]
        loss_values_of_preds = np.concatenate([train_losses, test_losses])
        distance_from_attack_threshold = {c: [] for c in range(self.num_classes)}
        for c in range(self.num_classes):
            distance_from_attack_threshold[c] = {alpha: [] for alpha in alphas}
        
        for c in range(self.num_classes):
            for alpha in alphas:
                filepath = f'{self.attack_results_dirpath}/attack_results_alpha_{alpha}_numdatinclass_{self.num_data_in_class}_class_{c}.npz'
                with np.load(filepath, allow_pickle=True) as data:
                    tp = data['tp'][()]
                    fp = data['fp'][()]
                    tn = data['tn'][()]
                    fn = data['fn'][()]
                    attack_threshold = data['per_class_thresholds'][()][c]
                tpr = tp / (tp + fn)
                fpr = fp / (fp + tn)
                tpr_values[c].append(tpr)
                fpr_values[c].append(fpr)
                for loss, class_label in zip(loss_values_of_preds, class_labels_of_preds):
                    if class_label == c:
                        distance_from_attack_threshold[c][alpha].append(np.abs(loss - attack_threshold))
                print(f"Plotting distance from attack threshold for alpha {alpha}...")
                alpha_str = str(alpha).replace('.', '_')
                fig, ax = plt.subplots()
                ax.hist(distance_from_attack_threshold[c][alpha], bins=75, alpha=0.5, color='b', edgecolor='b')
                ax.set_xlabel("Distance from Attack Threshold")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Alpha {alpha}")
                plt.savefig(f'{self.attack_results_dirpath}/distance_from_attack_threshold_alpha_{alpha_str}_class_{c}', dpi=250)
                plt.close(fig)

                # save data to CSV
                df = pd.DataFrame(data={
                'x': distance_from_attack_threshold[c][alpha]
                })
                df.to_csv(f'{self.attack_results_dirpath}/distance_from_attack_threshold_alpha_{alpha}_class_{c}_data.csv')

            tpr_values[c].insert(0, 0)
            fpr_values[c].insert(0, 0)
            tpr_values[c].append(1)
            fpr_values[c].append(1)

            auc_value[c] = round(auc(x=fpr_values[c], y=tpr_values[c]), 5)

            self.logger.critical(f"fpr_values for class {c}: {fpr_values[c]}, tpr_values for class {c}: {tpr_values[c]}, auc for class {c}: {auc_value[c]}")

            plt.plot(fpr_values[c],
                    tpr_values[c],
                    linewidth=2.0,
                    color='b',
                    label=f'AUC = {auc_value[c]}')
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.ylim([0.0, 1.1])
            plt.legend(loc='lower right')
            plt.savefig(f'{self.attack_results_dirpath}/tpr_vs_fpr_class_{c}', dpi=250)
            plt.close()

            # save data to CSV
            df = pd.DataFrame(data={
                'fpr': fpr_values[c],
                'tpr': tpr_values[c]
            })
            df.to_csv(f'{self.attack_results_dirpath}/tpr_vs_fpr_class_{c}_data.csv')

        

