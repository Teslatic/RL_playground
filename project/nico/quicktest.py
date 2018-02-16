#!/usr/bin/env python
import os
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

# def open_sweepReport(sweepReport):
#     print(sweepReport)
#     for parameter in sweepReport:
#         open_multiReport(sweepReport[parameter])
#         # print(read_sR[parameter])
#
# def open_multiReport(multiReport):
#     training_reports = []
#     test_reports = []
#     test_each = []
#     for report in range(len(multiReport)):
#         training_reports.append(multiReport[report][0])
#         test_reports.append(multiReport[report][1])
#         test_each.append(multiReport[report][2])
#     print(training_reports)
#     print(test_reports)
#     print(test_each)

plotter = Plotter('aba')
experiment_name = 'quicktest'
main_path = path.dirname(path.abspath(sys.modules['__main__'].__file__))
# exp_dir = create_experiment(main_path, experiment_name)
# read_sR = pickle.load(open('{}/sweepReport.p'.format(main_path), 'rb'))
# open_sweepReport(read_sR)
read_mR = pickle.load(open('{}/multiReport.p'.format(main_path), 'rb'))

# mean_vector, std_vector = plotter.calculate_test_statistics(read_mR)
# print(read_mR)
# print(mean_vector)
# print(std_vector)

# training_reports, test_reports, test_each = plotter.open_multiReport(read_mR)
# print(test_reports)
# create_plots_dir(main_path)
# plotter.plot_test_multireport(read_mR, main_path, 'multireport')
create_plots_dir(main_path)
read_sR = pickle.load(open('{}/sweepReport.p'.format(main_path), 'rb'))
plotter.plot_sweep_training(read_sR, main_path, 'sweep_plot_training')

# multireports = plotter.open_sweepReport(read_sR)
# print("This is the multi report after reading")
# print(multireports)
# meanofmeans, stdofstd = plotter.training_statistics_sweep(multireports)
# print('MEANS OF MEANS')
# print(meanofmeans)
# print('STD OF STD')
# print(stdofstd)

# print(read_sR)
# multireports = plotter.open_sweepReport(read_sR)
# print(multireports)
# print(len(multireports))
# print(multireports[0])
# for report in range(len(multireports)):
#     # print(multireports[report])
#     # print(multireports[report])
#     plotter.plot_test_multireport(multireports[report], main_path, 'multireport{}'.format(report))
        # read_sR[parameter][run]
# from assets.helperFunctions.conver import convert_vector2tensor
