'''
Created on Sep. 18, 2020
@author: hmo (hyunho.mo@unitn.it)
'''
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import sys
import random
import numpy as np
import json
from utils import network_launcher
from datetime import date
from network_training import network_train
from utils import network_launcher
# import tensorflow as tf
from scipy.stats import rankdata
import pandas as pd
today = date.today()
dt_string = today.strftime("%b-%d-%Y")

## load csv file (log of GA)
def rank_array_gen(data_array):
  temp_rmse = data_array.argsort()
  rank_array = np.empty_like(temp_rmse)
  rank_array[temp_rmse] = np.arange(len(data_array))
  return rank_array

fitness_mode_list = ["val_rmse", "aic", "combined"]
fitness_mode = 0
pop_size = 25 # toy example
n_generations = 25 # toy example
# Assign CSV file path & name
csv_filename = 'train_FD001.csv'  # user defined
csv_test_filename = 'test_FD001.csv'  # user defined
csv_rul_filename = 'RUL_FD001.txt'  # user defined
tmp_path = 'tmp/'
log_path = 'log/'
csv_path = tmp_path + csv_filename
RUL_FD_path = tmp_path + csv_rul_filename

log_file_path = log_path + 'log_%s_pop-%s_gen-%s.csv' % (fitness_mode_list[fitness_mode], pop_size, n_generations)
log_df = pd.read_csv(log_file_path)

df_num_rows = len(log_df.index)
log_df['idx'] = range(df_num_rows) + np.ones(df_num_rows)

# val_rmse_array = log_df['val_rmse_avg']
val_rmse_array = log_df['val_rmse']
aic_array = log_df['AIC']

val_rank_array = rank_array_gen(val_rmse_array)
aic_rank_array = rank_array_gen(aic_array)

log_df.insert(1, 'val_ranking', val_rank_array+np.ones(df_num_rows))
log_df.insert(2, 'aic_ranking', aic_rank_array+np.ones(df_num_rows))


# Assign the file path & name of the multi-head CNN-LSTM network

# Assign columns name
# The columns indicate the type of measured physical properties in csv file
# It can be an input argument for later use (application)
num_sensors = 26
test_engine_idx = 45
cols = ['unit_nr', 'cycles', 'os_1', 'os_2', 'os_3']
cols += ['sensor_{0:02d}'.format(s + 1) for s in range(num_sensors)]
cols_non_sensor = ['unit_nr', 'cycles', 'os_1', 'os_2', 'RUL']

# dropout=0.5
dropout=0

pd.options.mode.chained_assignment = None  # default='warn'

train_FD, cols_sensors = network_launcher().preprocessing_main(csv_filename, cols, cols_non_sensor)
test_FD, cols_sensors = network_launcher().preprocessing_main(csv_test_filename, cols, cols_non_sensor, train=False)

test_top_k = 10  # Test for top k individuals
test_loop_idx = 1  # Repeat the test for each individual
col_test_tag = ['test_rmse_1', 'test_rmse_2', 'test_rmse_3', 'test_rmse_4', 'test_rmse_5']
col_test_tag_score = ['test_score_1', 'test_score_2', 'test_score_3', 'test_score_4', 'test_score_5']
for idx in range(test_loop_idx):
    log_df[col_test_tag[idx]] = ''
log_df['test_rmse_avg'] = ''

for idx in range(test_loop_idx):
    log_df[col_test_tag_score[idx]] = ''
log_df['test_score_avg'] = ''

col_idx_rmse_test1 = log_df.columns.get_loc("test_rmse_1")
col_idx_rmse_test_avg = log_df.columns.get_loc("test_rmse_avg")
col_idx_score_test1 = log_df.columns.get_loc("test_score_1")
col_idx_score_test_avg = log_df.columns.get_loc("test_score_avg")

for idx in range(test_top_k):
    print("Test for top: ", idx + 1)
    top_k_row = log_df.loc[log_df['val_ranking'] == idx + 1]
    window_length = int(top_k_row['window_length'])
    n_filters = int(top_k_row['n_filters'])
    kernel_size = int(top_k_row['kernel_size'])
    n_conv_layer = int(top_k_row['n_conv_layer'])
    LSTM1_ref = int(top_k_row['LSTM1_units'] / top_k_row['n_window'])
    LSTM2_ref = int(top_k_row['LSTM2_units'] / top_k_row['n_window'])

    if csv_filename == 'train_FD001.csv':
        sequence_length = 30
        n_channel = 1
        strides_len = 1
        n_outputs = 1
        cross_val = False
        k_value_fold = 10
        val_split = 0.1
        patience = 5
        # max_epoch = int(top_k_row['stop_epoch'])
        max_epoch = 20
        bidirec = False
        stride = 1
        piecewise_lin_ref = 125
        batch_size = 400

    elif csv_filename == 'train_FD002.csv':
        sequence_length = 30
        n_channel = 1
        strides_len = 1
        n_outputs = 1
        cross_val = False
        k_value_fold = 10
        val_split = 0.2
        max_epoch = 28
        patience = 8
        bidirec = False
        stride = 1
        piecewise_lin_ref = 125
        batch_size = 400

    elif csv_filename == 'train_FD003.csv':
        sequence_length = 30
        n_channel = 1
        strides_len = 1
        n_outputs = 1
        cross_val = False
        k_value_fold = 10
        val_split = 0.2
        max_epoch = 40
        patience = 10
        bidirec = False
        stride = 1
        piecewise_lin_ref = 125
        batch_size = 400

    elif csv_filename == 'train_FD004.csv':
        sequence_length = 30
        n_channel = 1
        strides_len = 1
        n_outputs = 1
        cross_val = False
        k_value_fold = 10
        val_split = 0.2
        max_epoch = 40
        patience = 10
        bidirec = False
        stride = 1
        piecewise_lin_ref = 125
        batch_size = 400

    model_path =tmp_path +'trained_opt_model-wl_%s-nf_%s-ks_%s-nc_%s-l1_%s-l2_%s.h5' % (
    window_length, n_filters, kernel_size, n_conv_layer, LSTM1_ref, LSTM2_ref)

    training_input, training_input_label = network_launcher().opt_network_input_generator(
        dataframe_norm=train_FD,
        cols_non_sensor=cols_non_sensor,
        sequence_length=sequence_length,
        stride=stride,
        window_length=window_length,
        piecewise_lin_ref=piecewise_lin_ref)

    test_input, test_input_label = network_launcher().rmse_test_input_generator(
        dataframe_norm=test_FD,
        cols_non_sensor=cols_non_sensor,
        rul_file_path=RUL_FD_path,
        sequence_length=sequence_length,
        stride=stride,
        window_length=window_length,
        piecewise_lin_ref=piecewise_lin_ref
    )

    for loop in range(test_loop_idx):
        print("loop", loop)
        ## Generate the network and run the training and save the trained model into the file
        cnnlstm = network_launcher().network_training(training_input, training_input_label, cols_sensors, model_path,
                                                      fitness_mode, log_file_path,
                                                      test=False, test_engine_idx=None,
                                                      n_channel=1, n_filters=n_filters, strides_len=1,
                                                      kernel_size=kernel_size,
                                                      n_conv_layer=n_conv_layer, LSTM1_ref=LSTM1_ref,
                                                      LSTM2_ref=LSTM2_ref,
                                                      n_outputs=1, cross_val=False, k_value_fold=k_value_fold,
                                                      val_split=val_split,
                                                      batch_size=batch_size, max_epoch=max_epoch,
                                                      patience=patience, bidirec=False, dropout=0.5, experiment=True)

        rmse, score = network_train().opt_network_test_rmse(cnnlstm, test_input, test_input_label, model_path,
                                                            window_length,
                                                            n_filters, kernel_size, n_conv_layer, LSTM1_ref, LSTM2_ref)

        log_df[col_test_tag[loop]][log_df['val_ranking'] == idx + 1] = round(rmse, 2)
        log_df[col_test_tag_score[loop]][log_df['val_ranking'] == idx + 1] = round(score, 2)
        top_k_row[col_test_tag[loop]] = round(rmse, 2)
        top_k_row[col_test_tag_score[loop]] = round(score, 2)

        # print (log_df)

    test_rmse_list = top_k_row.iloc[:, col_idx_rmse_test1:col_idx_rmse_test_avg]
    log_df['test_rmse_avg'][log_df['val_ranking'] == idx + 1] = round(test_rmse_list.mean(1), 2)
    test_score_list = top_k_row.iloc[:, col_idx_score_test1:col_idx_score_test_avg]
    log_df['test_score_avg'][log_df['val_ranking'] == idx + 1] = round(test_score_list.mean(1), 2)

test_log_file_path = log_path + 'log_%s_pop-%s_gen-%s_test_deter_gpu.csv' % (
fitness_mode_list[fitness_mode], pop_size, n_generations)
log_df.to_csv(test_log_file_path, index=False)


def rank_array_gen(data_array):
  temp_rmse = data_array.argsort()
  rank_array = np.empty_like(temp_rmse)
  rank_array[temp_rmse] = np.arange(len(data_array))
  return rank_array

top_k_df = log_df.loc[log_df['val_ranking'] <= test_top_k]
df_num_rows= len(top_k_df.index)

test_rmse_rank_array = rank_array_gen(top_k_df['test_rmse_avg'])
test_score_rank_array = rank_array_gen(top_k_df['test_score_avg'])
top_k_df.insert(3, 'test_rmse_ranking', test_rmse_rank_array + np.ones(df_num_rows))
top_k_df.insert(4, 'test_score_ranking', test_score_rank_array + np.ones(df_num_rows))
test_log_file_path_top_k = log_path + 'log_%s_pop-%s_gen-%s_test_top_k_gpu.csv' % (fitness_mode_list[fitness_mode], pop_size, n_generations)
top_k_df.to_csv(test_log_file_path_top_k, index=False)
#
# test_log_file_path_top_k = log_path + 'log_%s_pop-%s_gen-%s_test_top_k_gpu.csv' % (fitness_mode_list[fitness_mode], pop_size, n_generations)
# top_k_df = pd.read_csv(test_log_file_path_top_k, header=0,  index_col=False)
# print (top_k_df)


top_k_df_sort = top_k_df.loc[np.argsort(top_k_df['val_ranking'])]
# print (top_k_df_sort)
# print (top_k_df_sort.columns)
x = top_k_df_sort['val_ranking'].to_numpy()
print (x)
y_1 = top_k_df_sort['val_rmse'].to_numpy()
y_2 = top_k_df_sort['test_rmse_avg'].to_numpy()
y_3 = top_k_df_sort['AIC'].to_numpy()

fig, ax1 = plt.subplots()
fig.set_size_inches(18.5, 10.5)

ax2 = ax1.twinx()
ax1.plot(x, y_1, 'r-.')
ax1.plot(x, y_2, 'g-o', linewidth=2)
ax2.plot(x, y_3, 'b--o')

ax1.set_xlabel('EA rank', fontsize=26)
ax1.tick_params(axis="x", labelsize=10)
ax1.set_ylabel('rmse_avg', color='r', fontsize=26)
ax2.set_ylabel('AIC', color='b', fontsize=26)
ax1.tick_params(axis="y", labelsize=20)
ax2.tick_params(axis="y", labelsize=20)
ax1.legend(['val_rmse', 'test_rmse'], loc='upper left', fontsize=20)
ax2.legend(['AIC'], loc='best', fontsize=20)
# ax1.set_yticks(fontsize=26)
# ax2.set_yticks(fontsize=26)
plt.xticks(x, fontsize=5)
ax1.plot([0, test_top_k], [11.94, 11.94], 'k-', lw=3,dashes=[2, 2])
ax1.text(test_top_k-2, 12, r'2020 SOTA', fontsize=20)
plt.show()

fig.savefig("graph/" +  'comparison_pop-%s_gen-%s_test_top_k_deter_gpu.png' % (pop_size, n_generations))








