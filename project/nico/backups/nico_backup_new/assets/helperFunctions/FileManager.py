import os
from os import path
from assets.helperFunctions.timestamps import print_timestamp

def create_experiment(dir_path, exp_name):
    """
    Creates the folder structure which is necessary to save files.
    """
    print_timestamp('Started experiment {}'.format(exp_name))
    exp_directory = dir_path + '/experiments/' + exp_name
    file_index = 0
    while path.exists(exp_directory + '%s' % file_index):
        file_index += 1

    idx_exp_dir = exp_directory + '{}'.format(file_index)
    os.makedirs(idx_exp_dir)
    os.makedirs(idx_exp_dir+'/results/policy_plots')
    os.makedirs(idx_exp_dir+'/results/parameters')
    os.makedirs(idx_exp_dir+'/results/plots/png')
    os.makedirs(idx_exp_dir+'/results/plots/pdf')
    os.makedirs(idx_exp_dir+'/results/logs')
    os.makedirs(idx_exp_dir+'/results/CSV')
    return idx_exp_dir
