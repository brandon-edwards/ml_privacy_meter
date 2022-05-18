#! /bin/bash
#$ -l h_vmem=100G ## amout RAM being requested
#$ -l gpu # request the more common P100 nodes
##$ -l A40 # to request the less common A40 nodes
## more details in https://sbia-wiki.uphs.upenn.edu/wiki/index.php/GPU_Computing
#$ -pe threaded 10 ## change number of CPU threads you want to request here
#$ -cwd
#$ -m b 
#$ -m e 
# this file is used to run gpu jobs on the cluster in a proper manner so 
# that the CUDA_VISIBLE_DEVICES environment variable is properly initialized
# ref: https://sbia-wiki.uphs.upenn.edu/wiki/index.php/GPU_Computing#Directing_Jobs_to_a_Specific_GPU_with_the_get_CUDA_VISIBLE_DEVICES_Utility
### $1: absolute path to python interpreter in virtual environment
### $2: absolute path to gandlf_run that needs to be invoked
### $3: absolute path to the data.csv file
### $1: yaml configuration
### $2: model path

## run actual trainer
/cbica/comp_space/patis/testing/gandlf_dp/venv/bin/python /cbica/home/patis/comp_space/testing/ml_privacy_meter/tutorials/population_attack_mnist.py \
-cp /cbica/home/patis/comp_space/testing/ml_privacy_meter/sbu_new_csv/SBU_pm_population_class_balanced.csv \
-ctr /cbica/home/patis/comp_space/testing/ml_privacy_meter/sbu_new_csv/SBU_pm_train_class_balanced.csv \
-cts /cbica/home/patis/comp_space/testing/ml_privacy_meter/sbu_new_csv/SBU_pm_test_class_balanced.csv \
-c $1 \
-m $2 
