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

plotter = Plotter('default')
# experiment_name = 'quicktest'
main_path = path.dirname(path.abspath(sys.modules['__main__'].__file__))
create_plots_dir(main_path)

read_sR = pickle.load(open('{}/sweepReport.p'.format(main_path), 'rb'))
# mR1 = pickle.load(open('{}/multiReport1.p'.format(main_path), 'rb'))
# mR2 = pickle.load(open('{}/multiReport2.p'.format(main_path), 'rb'))
# mR3 = pickle.load(open('{}/multiReport3.p'.format(main_path), 'rb'))
# mR4 = pickle.load(open('{}/multiReport4.p'.format(main_path), 'rb'))
# mR5 = pickle.load(open('{}/multiReport5.p'.format(main_path), 'rb'))
# mR6 = pickle.load(open('{}/multiReport6.p'.format(main_path), 'rb'))
# mR7 = pickle.load(open('{}/multiReport7.p'.format(main_path), 'rb'))

# print(mR1)
# mrtotal = mR1 + mR2 + mR3 + mR4 + mR5 + mR6
# print(mrtotal)
# plotter.plot_test_multireport(mrtotal, main_path, 'multireport')
# plotter.plot_test_multireport_clean(mrtotal, main_path, 'multireport')

# Sweepreports can be edited here

plotter.plot_training_sweep(read_sR, main_path, 'sweep_plot_training')
plotter.plot_test_sweep(read_sR, main_path, 'sweep_plot_test')



# *
# read_sR2 = pickle.load(open('{}/sweepReport2.p'.format(main_path), 'rb'))
# print(read_sR)
# print(read_sR2)
# read_sR.update(read_sR2)
# print(read_sR)


# create_plots_dir(main_path)
# read_mR = pickle.load(open('{}/multiReport.p'.format(main_path), 'rb'))
# # plotter.plot_test_multireport(read_mR, main_path, 'multireport')

# a = np.arange(100)
#
# for i in range(100):
#     print(random.randint(1, 10000))
