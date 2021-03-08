'''
Editor: @Hyunhomo
python --vesion:3.6.10
Reference: https://github.com/mohyunho
Reference: https://www.sciencedirect.com/science/article/abs/pii/S0925231219309877
'''

## Import libraries in python
'''
Please use below pip to install all package requirements
pip install -r py_pkg_requirements.txt
'''
import logging
import sys, os
import pandas as pd
import numpy as np
from datetime import date
from ts_preprocessing import ts_prep
from ts_window import ts_win
from network_training import network_train

class network_launcher(object):
    # Extract single input instance as the form of 4D tensor
    def __init__(self):
        '''
        Constructor
        @param none
        '''
        self.__logger = logging.getLogger("main of network part")

    def Extract(self, lst, sample_idx):
        extracted_input = []
        for sensor_idx in range(len(lst)):
            temp = lst[sensor_idx][sample_idx]
            temp = np.reshape(temp, (1, temp.shape[0], temp.shape[1], temp.shape[2]))
            extracted_input.append(temp)
        return extracted_input

    def preprocessing_main (self, csv_filename, cols, cols_non_sensor, train=True, piecewise_lin_ref=125):
        '''
        selecting columns in csv file, calculating RUL and all other kinds of data preparation
        :param csv_filename: user defined csv filename

        :return train_FD: save to the same csv file
        :return cols_sensors: selected sensor columns in dataframe that are used as the network input
        '''

        tmp_path = 'tmp/'
        csv_path = tmp_path + csv_filename
        print (csv_path)

        ## Load data from csv and assign to dataframe
        data_FD = ts_prep().load_train_csv(csv_path, cols)
        ## All processing within dataframe (RUL calculation, column drop and so on)
        data_FD = ts_prep().df_preparation(data_FD, train, piecewise_lin_ref)
        ## Normalize sensor measurement data
        data_FD, cols_sensors = ts_prep().df_preprocessing(data_FD, cols_non_sensor, train)


        print (data_FD)
        print ("pre-processing is successfully completed") # remark the end of the processing

        ## Return dataframe after preprocessing
        return data_FD, cols_sensors



    def network_input_generator (self, dataframe_norm, cols_non_sensor, test_engine_idx=None, train=True, application=False,
                                 rul_file_path=None, sequence_length = 30, stride = 1, window_length = 3,
                                 piecewise_lin_ref=125):
        '''
        generating the training input for multi-head CNN-LSTM network & the label of each input samples
        :param dataframe_norm: dataframe after preprocessing
        :param cols_non_sensor: columns in dataframe that are not the result of sensor measurement
        :param train: If it is true, then the func generate training input. Otherwise, the func generates test input
        :param application: If it is true, then the func generate tes input without ground truth (actual RUL).
        :param sequence_length: the number of time stamps for sliced time series
        :param stride: stride of ts window
        :param window_length: segmented time series from sequence
        :return network_input: ready to use input for CNN-LSTM network after applying time series window (python list)
        :return network_input_label: label(target value) of network input
        '''
        n_window = int((sequence_length - window_length) / (stride) + 1)
        n_engines = len(dataframe_norm['unit_nr'].unique())

        if train == True:
            ## Generate numpy array of sequence
            seq = ts_win().seq_generetion(dataframe_norm, cols_non_sensor,sequence_length)
            # print("the number of input samples: ", seq.shape[0])
            # print("the length of each sequence: ", seq.shape[1])
            ## Reshape the time series as the network input (python list of parallel input)
            network_input = ts_win().networkinput_generetion(seq, stride, n_window, window_length)
            ## Generate numpy array of label(target value) of each sequence
            network_input_label = ts_win().label_generetion(dataframe_norm, sequence_length)




        else:
            ts_array_test_engine = dataframe_norm[dataframe_norm['unit_nr'] == test_engine_idx+1]
            test_seq = ts_win().test_seq_generetion(ts_array_test_engine, cols_non_sensor, sequence_length)
            network_input = ts_win().networkinput_generetion(test_seq, stride, n_window, window_length)
            # print ("network_input", network_input[1].shape)

            if application == False:
                network_input_label = []

            else:
                ##Ground truth generator
                y_mask = [len(dataframe_norm[dataframe_norm['unit_nr'] == id]) >= sequence_length for id in
                          dataframe_norm['unit_nr'].unique()]
                # print ("y_mask", y_mask)
                col_rul = ['RUL_truth']
                RUL_FD = pd.read_csv(rul_file_path, sep=' ', header=None, names=col_rul, index_col=False)
                # Cut max RUL ground truth
                RUL_FD['RUL_truth'].loc[(RUL_FD['RUL_truth'] > piecewise_lin_ref)] = piecewise_lin_ref

                label_array_test_last = RUL_FD['RUL_truth'][y_mask].values
                label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)

                engine_RUL = label_array_test_last[test_engine_idx]
                test_seq_ground_truth = np.arange(test_seq.shape[0] + engine_RUL - 1, engine_RUL - 1, -1)
                test_seq_ground_truth[test_seq_ground_truth > piecewise_lin_ref] = piecewise_lin_ref
                network_input_label = test_seq_ground_truth


        return network_input, network_input_label


    def opt_network_input_generator (self, dataframe_norm, cols_non_sensor, sequence_length = 30,
                                     stride = 1, window_length = 3,piecewise_lin_ref=125):
        '''
        generating the training input for multi-head CNN-LSTM network & the label of each input samples
        :param dataframe_norm: dataframe after preprocessing
        :param cols_non_sensor: columns in dataframe that are not the result of sensor measurement
        :param train: If it is true, then the func generate training input. Otherwise, the func generates test input
        :param application: If it is true, then the func generate tes input without ground truth (actual RUL).
        :param sequence_length: the number of time stamps for sliced time series
        :param stride: stride of ts window
        :param window_length: segmented time series from sequence
        :return network_input: ready to use input for CNN-LSTM network after applying time series window (python list)
        :return network_input_label: label(target value) of network input
        '''
        n_window = int((sequence_length - window_length) / (stride) + 1)
        n_engines = len(dataframe_norm['unit_nr'].unique())


        ## Generate numpy array of sequence
        seq = ts_win().seq_generetion(dataframe_norm, cols_non_sensor,sequence_length)
        # print("the number of input samples: ", seq.shape[0])
        # print("the length of each sequence: ", seq.shape[1])
        ## Reshape the time series as the network input (python list of parallel input)
        network_input = ts_win().networkinput_generetion(seq, stride, n_window, window_length)
        ## Generate numpy array of label(target value) of each sequence
        network_input_label = ts_win().label_generetion(dataframe_norm, sequence_length)

        return network_input, network_input_label


    # def rmse_test_input_generator (self, test_FD, cols_non_sensor, test_engine_idx=None,
    #                              rul_file_path=None, sequence_length = 30, stride = 1, window_length = 3,
    #                              piecewise_lin_ref=125):
    #     '''
    #     generating the training input for multi-head CNN-LSTM network & the label of each input samples
    #     :param dataframe_norm: dataframe after preprocessing
    #     :param cols_non_sensor: columns in dataframe that are not the result of sensor measurement
    #
    #     '''
    #     n_window = int((sequence_length - window_length) / (stride) + 1)
    #
    #
    #     sequence_cols_test = test_FD.columns.difference(cols_non_sensor)
    #     # We pick the last sequence for each id in the test data
    #     seq_array_test_last = [test_FD[test_FD['unit_nr'] == id][sequence_cols_test].values[-sequence_length:]
    #                            for id in test_FD['unit_nr'].unique() if
    #                            len(test_FD[test_FD['unit_nr'] == id]) >= sequence_length]
    #     # print(seq_array_test_last[0].shape)
    #
    #     seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
    #     # print("seq_array_test_last")
    #     # print(seq_array_test_last.shape)
    #
    #     # Reshape test array to 4D tensor (as the input for CNN-LSTM)
    #     network_input = ts_win().networkinput_generetion(seq_array_test_last, stride, n_window, window_length)
    #     # print ("network_input", network_input[1].shape)
    #
    #     ##Ground truth generator
    #     y_mask = [len(test_FD[test_FD['unit_nr'] == id]) >= sequence_length for id in
    #               test_FD['unit_nr'].unique()]
    #     # print("y_mask", y_mask)
    #
    #     col_rul = ['RUL_truth']
    #     RUL_FD = pd.read_csv(rul_file_path, sep=' ', header=None, names=col_rul, index_col=False)
    #     # Cut max RUL ground truth
    #     RUL_FD['RUL_truth'].loc[(RUL_FD['RUL_truth'] > piecewise_lin_ref)] = piecewise_lin_ref
    #     label_array_test_last = RUL_FD['RUL_truth'][y_mask].values
    #     network_input_label = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)
    #     # print(network_input_label)
    #     # print(network_input_label.shape)
    #
    #     return network_input, network_input_label

    def rmse_test_input_generator (self, dataframe_norm, cols_non_sensor, test_engine_idx=None,
                                 rul_file_path=None, sequence_length = 30, stride = 1, window_length = 3,
                                 piecewise_lin_ref=125):
        '''
        generating the training input for multi-head CNN-LSTM network & the label of each input samples
        :param dataframe_norm: dataframe after preprocessing
        :param cols_non_sensor: columns in dataframe that are not the result of sensor measurement

        '''
        n_window = int((sequence_length - window_length) / (stride) + 1)
        n_engines = len(dataframe_norm['unit_nr'].unique())

        sequence_cols_test = dataframe_norm.columns.difference(cols_non_sensor)
        # We pick the last sequence for each id in the test data
        seq_array_test_last = [dataframe_norm[dataframe_norm['unit_nr'] == id][sequence_cols_test].values[-sequence_length:]
                               for id in dataframe_norm['unit_nr'].unique() if
                               len(dataframe_norm[dataframe_norm['unit_nr'] == id]) >= sequence_length]
        # print(seq_array_test_last[0].shape)

        seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
        # print("seq_array_test_last")
        # print(seq_array_test_last.shape)

        # Reshape test array to 4D tensor (as the input for CNN-LSTM)
        network_input = ts_win().networkinput_generetion(seq_array_test_last, stride, n_window, window_length)
        # print ("network_input", network_input[1].shape)

        ##Ground truth generator
        y_mask = [len(dataframe_norm[dataframe_norm['unit_nr'] == id]) >= sequence_length for id in
                  dataframe_norm['unit_nr'].unique()]
        # print("y_mask", y_mask)

        col_rul = ['RUL_truth']
        RUL_FD = pd.read_csv(rul_file_path, sep=' ', header=None, names=col_rul, index_col=False)
        # Cut max RUL ground truth
        RUL_FD['RUL_truth'].loc[(RUL_FD['RUL_truth'] > piecewise_lin_ref)] = piecewise_lin_ref
        label_array_test_last = RUL_FD['RUL_truth'][y_mask].values
        network_input_label = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)
        # print(network_input_label)
        # print(network_input_label.shape)

        return network_input, network_input_label



    def network_training (self, training_input, training_input_label, cols_sensors, model_path, fitness_mode, log_file_path,
                          dropout, test=False, test_engine_idx=None,
                          n_channel=1, n_filters=2 , strides_len=1, kernel_size=3, n_conv_layer=2, LSTM1_ref=10, LSTM2_ref=5,
                          n_outputs=1, cross_val=True, k_value_fold=10, val_split=0.2, batch_size=400, max_epoch=40,
                          patience=10, bidirec = False, experiment = False, geno_list =None):
        '''
        generating the multi-head CNN-LSTM network & training the network
        :param training_input: ready to use input for CNN-LSTM network after applying time series window (python list)
        :param training_input_label: label(target value) of network input
        :param cols_sensors: selected sensor columns in dataframe that are used as the network input
        :param cols_sensors: selected sensor columns in dataframe that are used as the network input
        :param model_path: h5 file path of the network model
        :param n_channel: the number of network channel (in the case of multi-head, it should be 1)
        :param n_filters: the number of convolutional filters, increase the number
        if the network cannot recognize the pattern
        :param strides_len: stride length of convolution
        :param kernel_size: kernel size of convolution
        :param n_conv_layer: the number of (stacked) convolutional layers, increase the number
        if the network cannot recognize the pattern. Max number is 4
        :param LSTM1_ref: reference number for the number of units in LSTM layer 1
        :param LSTM2_ref: reference number for the number of units in LSTM layer 2
        :param n_outputs: the number of output of the network (in the case of RUL prediction, it should be 1)
        :param cross_val: whether to use k-fold cross validation
        (please consider cross validation only the case of using GPU)
        :param k_value_fold: k-value for fold cross validation (10 or 5 is recommended)
        :param val_split: the ratio of validation (from training) for the decision of early stopping
        :param batch_size: the batch size for optimization (gradient)
        :param max_epoch: the maximum number of training epoch (if the training loss does not converge
        without early stopping, increase the number)
        :param patience: epoch patience for early stopping decision
        :param bidirec: whether to use bidirectional for LSTM layers

        :return trained network is saved to model path
        '''
        ## Defining relative parameters
        n_training_samples = training_input[0].shape[0]
        n_window = training_input[0].shape[1]
        window_length = training_input[0].shape[2]
        # LSTM_u1 = n_window*LSTM1_ref
        # LSTM_u2 = n_window*LSTM2_ref
        # LSTM_u1 = LSTM1_ref
        # LSTM_u2 = LSTM2_ref
        LSTM_u1 = 20*LSTM1_ref
        LSTM_u2 = 20*LSTM2_ref

        ## Training the network
        if test == False:
            fitness_net = network_train().network_training(training_input, training_input_label,
                                            cols_sensors, model_path, fitness_mode, log_file_path,
                                             LSTM_u1, LSTM_u2, cross_val, k_value_fold, val_split, n_channel,
                                             n_filters, strides_len, kernel_size, window_length, n_window,
                                             n_conv_layer, n_outputs, batch_size, max_epoch, patience, bidirec, dropout, experiment, geno_list)
            # print ("log_file_path", log_file_path)
            # temp_df.to_csv(log_file_path, mode='a', header=None)

            return fitness_net




        elif test == True:
            y_predicted, y_actual = network_train().network_test(training_input, training_input_label, test_engine_idx,
                                                     model_path)
            return y_predicted, y_actual



