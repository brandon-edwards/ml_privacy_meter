import pandas as pd 
import os
import numpy as np
from pathlib import Path

"""
This script takes train and val csvs and splits them in order to create population, train, and test csvs to be used
for privacy meter evaluation. The old test set is split into privacy meter population and test sets. A privacy meter train set is then
created by pulling from the original train set in such a way that the final privacy meter train and test sets are equal. 

"""

#######################################
# Here are some hard coded choices
######################################

## these should come from cli
orig_train_csv_path = "/cbica/home/patis/comp_space/testing/gandlf_dp_experiments/train.csv"
orig_test_csv_path = "/cbica/home/patis/comp_space/testing/gandlf_dp_experiments/valid.csv"
new_csv_folder = "/cbica/home/patis/comp_space/testing/ml_privacy_meter/sbu_new_csv"
Path(new_csv_folder).mkdir(parents=True, exist_ok=True)

data_name = "SBU"

# original test samples will be split into a population set, a test set, and possibly a set that gets dropped
orig_test_portion_to_pop = 0.5

allow_dropped_orig_test_samples = False
allow_dropped_orig_train_samples = True

shuffle = True

#######################################
#######################################
if __name__ == '__main__':
    orig_train_df = pd.read_csv(orig_train_csv_path)
    orig_test_df = pd.read_csv(orig_test_csv_path)


    # shuffle rows if specified
    if shuffle:
        orig_train_df = orig_train_df.sample(frac=1).reset_index(drop=True)
        orig_test_df = orig_test_df.sample(frac=1).reset_index(drop=True)
    
    # column names are now case-insensitive
    orig_train_df.columns = orig_train_df.columns.str.lower()
    # now split each by class to preserve the class balance when you sample
    classes = list(orig_train_df['valuetopredict'].unique())
    print(20*"#")
    print(f"Detecting the complete class list from the train csv and found: {classes}.")
    print()

    per_class_orig_train_df_dict = {_class: orig_train_df[orig_train_df['valuetopredict']==_class] for _class in classes}
    orig_train_perclass_counts = [len(this_dict) for this_dict in per_class_orig_train_df_dict.values()]

    per_class_orig_test_df_dict = {_class: orig_test_df[orig_test_df['valuetopredict']==_class] for _class in classes} 
    orig_test_perclass_counts = [len(this_dict) for this_dict in per_class_orig_test_df_dict.values()]
    


    nb_orig_test = len(orig_test_df)
    nb_orig_train = len(orig_train_df)


    nb_pop = int(orig_test_portion_to_pop * nb_orig_test)
    frac_test_to_pop = float(nb_pop)/float(nb_orig_test)


    if (nb_orig_test - nb_pop < nb_orig_train):
        if not allow_dropped_orig_train_samples:
            raise ValueError(f"{nb_orig_test - nb_pop} samples will become testing samples which will require dropping original training"\
                         f" samples as there are {nb_orig_train}, but allow_dropped_train_samples is False.")
        else:
            nb_test = nb_train = nb_orig_test - nb_pop
            frac_train_to_train = float(nb_train)/float(nb_orig_train)

            
            #pull off the samples needed
            cutpoints = {_class: int(np.floor(frac_test_to_pop * orig_test_perclass_counts[_class])) for _class in classes}
            print("CUTPOINTS ARE: ", cutpoints)
            pop_df = pd.concat([per_class_orig_test_df_dict[_class][:cutpoints[_class]] for _class in classes], axis=0)

            test_df = pd.concat([per_class_orig_test_df_dict[_class][cutpoints[_class]:] for _class in classes], axis=0)

            train_df = pd.concat([per_class_orig_train_df_dict[_class][:int(np.floor(orig_train_perclass_counts[_class]*frac_train_to_train))] for _class in classes], axis=0)
# working here continue to make a stratifiec split TODO
    elif (nb_orig_test - nb_pop > nb_orig_train):
        if not allow_dropped_orig_test_samples:
            raise ValueError(f"{nb_orig_test - nb_pop} samples will become testing samples which will require dropping original testing"\
                         f" samples as there are {nb_orig_test}, but allow_dropped_test_samples is False.")
        else:
            nb_test = nb_train = nb_orig_train
            
            frac_test_to_test = float(nb_test)/float(nb_orig_test)
            second_cutpoint_deltas = {_class: int(np.floor(frac_test_to_test * orig_test_perclass_counts[_class])) for _class in classes}

            
            #pull off the samples needed
            cutpoints = {_class: int(np.floor(frac_test_to_pop * orig_test_perclass_counts[_class])) for _class in classes}
            pop_df = pd.concat([per_class_orig_test_df_dict[_class][:cutpoints[_class]] for _class in classes], axis=0)

            second_cutpoints = {_class: cutpoints + second_cutpoint_deltas[_class] for _class in classes}

            test_df = pd.concat([per_class_orig_test_df_dict[_class][cutpoints[_class]:(cutpoints[_class]+second_cutpoint_deltas[_class])] for _class in classes], axis=0)

            train_df = orig_train_df

    else:
        nb_test = nb_train = nb_orig_train

        # pull off the samples needed
        cutpoints = {_class: int(np.floor(frac_test_to_pop * orig_test_perclass_counts[_class])) for _class in classes}
        pop_df = pd.concat([per_class_orig_test_df_dict[_class][:cutpoints[_class]] for _class in classes], axis=0)
        
        test_df = pd.concat([per_class_orig_test_df_dict[_class][cutpoints[_class]:] for _class in classes], axis=0)
        train_df = orig_train_df
    
    if not (0.9 < len(test_df) / len(train_df) < 1.1):
        raise ValueError(f"Test and train did not end up within 1 percent of eachother, got {len(test_df)} and {len(train_df)} respectively.")
    if len(test_df) + len(pop_df) != len(orig_test_df):
        raise ValueError(f"Test plus train lengths: {len(test_df) + len(pop_df)} does not equal orig test length: {len(orig_test_df)}")

    new_train_csv_path = os.path.join(new_csv_folder, data_name + "_pm_train_class_balanced.csv") 
    new_test_csv_path = os.path.join(new_csv_folder, data_name + "_pm_test_class_balanced.csv") 
    new_pop_csv_path = os.path.join(new_csv_folder, data_name + "_pm_population_class_balanced.csv")

    train_df.to_csv(new_train_csv_path, index=False)
    test_df.to_csv(new_test_csv_path, index=False)
    pop_df.to_csv(new_pop_csv_path, index=False)
