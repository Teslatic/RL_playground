#!/usr/bin/env python
import os
import random
from os import path
import sys
if "../" not in sys.path:
    sys.path.append("../")
# if "../" not in sys.path:
#     sys.path.append("../")

import numpy as np
from lib.envs.pendulum import PendulumEnv
import pickle
from assets.helperFunctions.FileManager import create_experiment
from assets.helperFunctions.FileManager import create_plots_dir
from assets.plotter.DankPlotters import Plotter

a = np.arange(100)

for i in range(100):
    print(random.randint(1, 10000))

# plotter = Plotter('aba')
# experiment_name = 'quicktest'
# main_path = path.dirname(path.abspath(sys.modules['__main__'].__file__))
#
# # create_plots_dir(main_path)
# read_mR = pickle.load(open('{}/multiReport.p'.format(main_path), 'rb'))
# # plotter.plot_test_multireport(read_mR, main_path, 'multireport')
#
# create_plots_dir(main_path)
#
# read_sR = pickle.load(open('{}/sweepReport.p'.format(main_path), 'rb'))
# # *
# plotter.plot_training_sweep(read_sR, main_path, 'sweep_plot_training')
# plotter.plot_test_sweep(read_sR, main_path, 'sweep_plot_test')



# *
# read_sR2 = pickle.load(open('{}/sweepReport2.p'.format(main_path), 'rb'))
# print(read_sR)
# print(read_sR2)
# read_sR.update(read_sR2)
# print(read_sR)
