#!/bin/python3
"""
Launcher for the experiments
Author: Leonardo Lucio Custode
Date: 17/09/2020
"""
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import sys
import random
import numpy as np
import json
from datetime import date
from utils import network_launcher
from task import SimpleNeuroEvolutionTask
from evolutionary_algorithm import GeneticAlgorithm
import time
import pandas as pd
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)
# tf.get_logger().setLevel('INFO')

# import multiprocess.context as ctx
# ctx._force_start_method('spawn')

# import tensorflow as tf
#
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)
experiment = False

# Check system path and date
print(sys.path)
today = date.today()
dt_string = today.strftime("%b-%d-%Y")

subdata_mode_list = ["fd001", "fd002", "fd003", "fd004"]
subdata_mode = 3
trial=3


# Assign CSV file path & name
if subdata_mode == 0:
    csv_filename = 'train_FD001.csv'  # user defined
    csv_test_filename = 'test_FD001.csv'  # user defined
    csv_rul_filename = 'RUL_FD001.txt'  # user defined
    # Other parameters
    # window_length = None #GA parameter
    # n_window = None #Internal variable calculated by n_window = int((sequence_length - window_length) / (stride) + 1)
    sequence_length = 30
    n_channel = 1
    strides_len = 1
    n_outputs = 1
    cross_val = False
    k_value_fold = 10
    val_split = 0.1
    max_epoch = 20
    patience = 20
    bidirec = False
    stride = 1
    piecewise_lin_ref = 125
    batch_size = 400
    dropout = 0

    seed = 0
    individual_seed = [[3, 2, 2, 15, 8]]



elif subdata_mode == 1:
    csv_filename = 'train_FD002.csv'  # user defined
    csv_test_filename = 'test_FD002.csv'  # user defined
    csv_rul_filename = 'RUL_FD002.txt'  # user defined
    # Other parameters
    # window_length = None #GA parameter
    # n_window = None #Internal variable calculated by n_window = int((sequence_length - window_length) / (stride) + 1)
    sequence_length = 21
    n_channel = 1
    strides_len = 1
    n_outputs = 1
    cross_val = False
    k_value_fold = 10
    val_split = 0.1
    max_epoch = 20
    patience = 20
    bidirec = False
    stride = 1
    piecewise_lin_ref = 125
    batch_size = 400
    dropout = 0

    seed = 0
    individual_seed = [4, 4, 1, 18, 9]



elif subdata_mode == 2:
    csv_filename = 'train_FD003.csv'  # user defined
    csv_test_filename = 'test_FD003.csv'  # user defined
    csv_rul_filename = 'RUL_FD003.txt'  # user defined
    # Other parameters
    # window_length = None #GA parameter
    # n_window = None #Internal variable calculated by n_window = int((sequence_length - window_length) / (stride) + 1)
    sequence_length = 38
    n_channel = 1
    strides_len = 1
    n_outputs = 1
    cross_val = False
    k_value_fold = 10
    val_split = 0.1
    max_epoch = 20
    patience = 20
    bidirec = False
    stride = 1
    piecewise_lin_ref = 125
    batch_size = 400
    dropout = 0

    seed = 0
    individual_seed = [3, 2, 2, 15, 8]



elif subdata_mode == 3:
    csv_filename = 'train_FD004.csv'  # user defined
    csv_test_filename = 'test_FD004.csv'  # user defined
    csv_rul_filename = 'RUL_FD004.txt'  # user defined
    # Other parameters
    # window_length = None #GA parameter
    # n_window = None #Internal variable calculated by n_window = int((sequence_length - window_length) / (stride) + 1)
    sequence_length = 19
    n_channel = 1
    strides_len = 1
    n_outputs = 1
    cross_val = False
    k_value_fold = 10
    val_split = 0.1
    max_epoch = 20
    patience = 20
    bidirec = False
    stride = 1
    piecewise_lin_ref = 125
    batch_size = 400
    dropout = 0

    seed = 0
    individual_seed = [3,3,2,10,5]




tmp_path = 'tmp/'
log_path = 'log/'
csv_path = tmp_path + csv_filename
RUL_FD_path = tmp_path + csv_rul_filename
print("csv_path",csv_path)
# Assign the file path & name of the multi-head CNN-LSTM network
model_path = tmp_path + 'file_%s_dt_%s.h5' % (csv_filename, dt_string)

# Assign columns name
# The columns indicate the type of measured physical properties in csv file
# It can be an input argument for later use (application)
num_sensors = 26
test_engine_idx = 45
cols = ['unit_nr', 'cycles', 'os_1', 'os_2', 'os_3']
cols += ['sensor_{0:02d}'.format(s + 1) for s in range(num_sensors)]
cols_non_sensor = ['unit_nr', 'cycles', 'os_1', 'os_2', 'RUL']

fitness_mode_list = ["val_rmse", "aic", "val_score", "rmse_combined", "score_combined"]
fitness_mode = 2






train_FD, cols_sensors = network_launcher().preprocessing_main(csv_filename, cols, cols_non_sensor)
test_FD, cols_sensors = network_launcher().preprocessing_main(csv_test_filename, cols, cols_non_sensor, train=False)



## Parameters for the GA
pop_size = 50 # toy example
n_generations = 50 # toy example
cx_prob = 0.5 # 0.25
mut_prob = 0.5 # 0.7
cx_op = "one_point"
mut_op = "uniform"
sel_op = "best"
other_args = {
    'mut_gene_probability': 0.3 #0.1
}


jobs = 2

random.seed(seed)
np.random.seed(seed)


start = time.time()


log_file_path = log_path + 'log_%s_%s_pop-%s_gen-%s_%s.csv' % (subdata_mode_list[subdata_mode], fitness_mode_list[fitness_mode], pop_size, n_generations, trial)
log_col = ['idx', 'stop_epoch', 'window_length', 'n_filters' , 'kernel_size' , 'n_conv_layer' , 'LSTM1_units' ,
           'LSTM2_units', 'n_window',
           'val_rmse', 'val_score', 'rmse_combined', 'score_combined',
           'AIC', 'train_loss', 'mle_term', 'params_term','geno_list']
log_df = pd.DataFrame(columns=log_col, index=None)
log_df.to_csv(log_file_path, index=False)
print (log_df)

mutate_log_path = 'EA_log/mute_log_%s_%s_%s_%s_%s.csv' % (subdata_mode_list[subdata_mode], fitness_mode_list[fitness_mode], pop_size, n_generations, trial)
mutate_log_col = ['idx', 'params_1','params_2','params_3','params_4','params_5', 'fitness', 'gen']
mutate_log_df = pd.DataFrame(columns=mutate_log_col, index=None)
mutate_log_df.to_csv(mutate_log_path, index=False)

def log_function(population, gen, mutate_log_pat=mutate_log_path):
    for i in range(len(population)):
        if population[i]==[]:
            "non_mutated empty"
            pass
        else :
            # print ("i: ", i)
            population[i].append(population[i].fitness.values[0])
            population[i].append(gen)

    temp_df = pd.DataFrame(np.array(population), index=None)
    temp_df.to_csv(mutate_log_path, mode='a', header=None)
    print ("population saved")
    return


task = SimpleNeuroEvolutionTask(
    dataframe_norm=train_FD,
    cols_non_sensor=cols_non_sensor,
    cols_sensors=cols_sensors,
    model_path=model_path,
    fitness_mode=fitness_mode,
    log_file_path=log_file_path,
    dropout=dropout,
    # n_window=n_window,
    sequence_length=sequence_length,
    n_channel=n_channel,
    strides_len=strides_len,
    n_outputs=n_outputs,
    cross_val=cross_val,
    k_value_fold=k_value_fold,
    val_split=val_split,
    max_epoch=max_epoch,
    patience=patience,
    bidirec=bidirec,
    test_engine_idx=test_engine_idx,
    stride=stride,
    piecewise_lin_ref=piecewise_lin_ref,
    batch_size=batch_size,
    experiment=experiment
)


# aic = task.evaluate(individual_seed)


ga = GeneticAlgorithm(
    task=task,
    population_size=pop_size,
    n_generations=n_generations,
    cx_probability=cx_prob,
    mut_probability=mut_prob,
    crossover_operator=cx_op,
    mutation_operator=mut_op,
    selection_operator=sel_op,
    seed=individual_seed,
    jobs=jobs,
    log_function = log_function,
    **other_args
)

pop, log, hof = ga.run()



print("Best individual:")
print(hof[0])

# Save to the txt file
# hof_filepath = tmp_path + "hof/best_params_fn-%s_ps-%s_ng-%s.txt" % (csv_filename, pop_size, n_generations)
# with open(hof_filepath, 'w') as f:
#     f.write(json.dumps(hof[0]))

print("Best individual is saved")
end = time.time()
print(end - start)

