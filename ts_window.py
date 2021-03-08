'''
Created on August 17, 2020
@author: hmo (hyunho.mo@unitn.it)
'''
import logging
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing

class ts_win(object):
    '''
    class for time series window application
    '''

    def __init__(self):
        '''
        Constructor
        @param none
        '''
        self.__logger = logging.getLogger('application of time series window for preparing network input')



    ## function to reshape features into (samples, time steps, features)
    @staticmethod
    def gen_sequence(id_df, seq_length, seq_cols):
        """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
        we need to drop those which are below the window-length. An alternative would be to pad sequences so that
        we can use shorter ones """
        # for one id I put all the rows in a single matrix
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        # Iterate over two lists in parallel.
        # For example id1 have 192 rows and sequence_length is equal to 50
        # so zip iterate over two following list of numbers (0,142),(50,192)
        # 0 50 -> from row 0 to row 50
        # 1 51 -> from row 1 to row 51
        # 2 52 -> from row 2 to row 52
        # ...
        # 142 192 -> from row 142 to 192
        for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
            yield data_matrix[start:stop, :]

    ## function to generate labels for training
    @staticmethod
    def gen_labels(id_df, seq_length, label):
        """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
        we need to drop those which are below the window-length. An alternative would be to pad sequences so that
        we can use shorter ones """
        # For one id I put all the labels in a single matrix.
        # For example:
        # [[1]
        # [4]
        # [1]
        # [5]
        # [9]
        # ...
        # [200]]
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]
        # I have to remove the first seq_length labels
        # because for one id the first sequence of seq_length size have as target
        # the last label (the previus ones are discarded).
        # All the next id's sequences will have associated step by step one label as target.
        return data_matrix[seq_length:num_elements, :]


    def seq_generetion(self, train_FD_norm, cols_non_sensor,sequence_length):
        '''
        :param train_FD_norm: path of csv file
        :param cols_non_sensor: declared columns in csv
        :return: numpy array of sequence (sliced time series)
        '''
        # pick the feature columns
        sequence_cols_train = train_FD_norm.columns.difference(cols_non_sensor)
        ## generator for the sequences
        # transform each id of the train dataset in a sequence
        seq_gen = (list(ts_win.gen_sequence(train_FD_norm[train_FD_norm['unit_nr'] == id], sequence_length, sequence_cols_train))
        for id in train_FD_norm['unit_nr'].unique())

        # generate sequences and convert to numpy array
        seq_array_train = np.concatenate(list(seq_gen)).astype(np.float32)

        return seq_array_train



    def test_seq_generetion(self, ts_array_test_engine, cols_non_sensor,sequence_length):
        '''
        :param ts_array_test_engine: time series array of the selected test engine
        :param cols_non_sensor: declared columns in csv
        :return: numpy array of sequence (sliced time series)
        '''
        # pick the feature columns
        sequence_cols_test = ts_array_test_engine.columns.difference(cols_non_sensor)
        ## generator for the sequences
        test_seq_gen = (list(ts_win.gen_sequence(ts_array_test_engine, sequence_length, sequence_cols_test)))
        print(test_seq_gen[0].shape)
        seq_array_test_engine = np.stack(list(test_seq_gen), axis=0).astype(np.float32)

        return seq_array_test_engine



    def label_generetion(self, train_FD_norm, sequence_length):
        '''
        :param train_FD_norm: path of csv file
        :return: numpy array of sequence (sliced time series)
        '''
        label_gen = [ts_win.gen_labels(train_FD_norm[train_FD_norm['unit_nr'] == id], sequence_length, ['RUL'])
                     for id in train_FD_norm['unit_nr'].unique()]

        label_array_train = np.concatenate(label_gen).astype(np.float32)

        return label_array_train


    def networkinput_generetion(self, seq_array_train, stride, n_window, window_length):
        '''
        :param numpy array of sequence (sliced time series)
        :return: numpy array of network input for training
        '''
        # for each sensor: reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
        train_FD_sensor = []

        as_strided = np.lib.stride_tricks.as_strided

        for s_i in range(seq_array_train.shape[2]):
            window_list = []
            window_array = np.array([])

            for seq in range(seq_array_train.shape[0]):
                S = stride
                s0 = seq_array_train[seq, :, s_i].strides
                seq_sensor = as_strided(seq_array_train[seq, :, s_i], (n_window, window_length),
                                        strides=(S * s0[0], s0[0]))
                #         print (seq_sensor)
                #         window_array = np.concatenate((window_array, seq_sensor), axis=1)
                window_list.append(seq_sensor)

            window_array = np.stack(window_list, axis=0)
            window_array = np.reshape(window_array,
                                      (window_array.shape[0], window_array.shape[1], window_array.shape[2], 1))
            # print(window_array.shape)
            train_FD_sensor.append(window_array)

        return train_FD_sensor