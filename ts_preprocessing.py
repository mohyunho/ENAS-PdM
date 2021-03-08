'''
Created on August 16, 2020
@author: hmo (hyunho.mo@unitn.it)
'''
import logging
import pandas as pd
import numpy as np
import sklearn as sk
import joblib
from sklearn import preprocessing


class ts_prep(object):
    '''
    class for preprocessing and data preparation
    '''

    def __init__(self):
        '''
        Constructor
        @param none
        '''
        self.__logger = logging.getLogger('data preparation for using it as the network input')


    @staticmethod
    def dummy(input):
        """
        static method
        """

        return input

    def load_train_csv(self, data_path_list, columns_ts):
        '''
        :param data_path_list: path of csv file
        :param columns_ts: declared columns in csv
        :return: assigned pandas dataframe from csv
        '''
        train_FD = pd.read_csv(data_path_list, sep=' ', header=None, names=columns_ts, index_col=False)

        # Checks on the .csv columns to see if the data for the training are correctly structured
        csv_raw = pd.read_csv(data_path_list, sep=' ', header=None, index_col=False)
        if len(train_FD.columns) == len(csv_raw.columns):
            pass
        elif len(train_FD.columns) > len(csv_raw.columns):
            print ('# of columns (%s) > # of sensors (%s), check the number of used sensors and columns' %(len(train_FD.columns), len(csv_raw.columns)))
        elif len(train_FD.columns) < len(csv_raw.columns):
            raise ValueError('# of columns (%s) < # of sensors (%s), check the number of used sensors and columns' %(len(train_FD.columns), len(csv_raw.columns)))

        return train_FD

    def df_preparation(self, train_FD, train, piecewise_lin_ref):
        '''
        Calculating RUL and Applying piecewise linear and column drop and so on
        :param train_FD: assigned pandas dataframe from csv
        :param piecewise_lin_ref: piecewise linear reference
        :return train_FD: dataframe after prep
        '''
        ## Caculate RUL and apply piecewise linear
        mapper = {}
        for unit_nr in train_FD['unit_nr'].unique():
            mapper[unit_nr] = train_FD['cycles'].loc[train_FD['unit_nr'] == unit_nr].max()

        # calculate RUL = time.max() - time_now for each unit
        if train == True:
            train_FD['RUL'] = train_FD['unit_nr'].apply(lambda nr: mapper[nr]) - train_FD['cycles']
            # piecewise linear for RUL labels
            train_FD['RUL'].loc[(train_FD['RUL'] > piecewise_lin_ref)] = piecewise_lin_ref
        else:
            pass

        ## Excluse columns which only have NaN as value
        # nan_cols = ['sensor_{0:02d}'.format(s + 22) for s in range(5)]
        cols_nan = train_FD.columns[train_FD.isna().any()].tolist()
        # print('Columns with all nan: \n' + str(cols_nan) + '\n')
        cols_const = [col for col in train_FD.columns if len(train_FD[col].unique()) <= 2]
        # print('Columns with all const values: \n' + str(cols_const) + '\n')

        ## Drop exclusive columns
        # train_FD = train_FD.drop(columns=cols_const + cols_nan)
        # test_FD = test_FD.drop(columns=cols_const + cols_nan)

        train_FD = train_FD.drop(columns=cols_const + cols_nan + ['sensor_01', 'sensor_05', 'sensor_06',
                                                                  'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'])



        return train_FD


    def df_preprocessing(self, df, cols_non_sensor, train, scaler_filename="scaler.save"):
        '''
        :param df: prepared dataframe
        :return df: dataframe (normalized values) after preprocessing
        :return cols_normalize.tolist(): selected sensor columns in dataframe that are used as the network input
        '''
        cols_normalize = df.columns.difference(cols_non_sensor)
        min_max_scaler = preprocessing.MinMaxScaler()
        if train == True:
            norm_df = pd.DataFrame(min_max_scaler.fit_transform(df[cols_normalize]),
                               columns=cols_normalize,
                               index=df.index)

            joblib.dump(min_max_scaler, scaler_filename)

        else:
            min_max_scaler = joblib.load(scaler_filename)
            norm_df = pd.DataFrame(min_max_scaler.transform(df[cols_normalize]),
                               columns=cols_normalize,
                               index=df.index)
        join_df = df[df.columns.difference(cols_normalize)].join(norm_df)
        df = join_df.reindex(columns=df.columns)

        return df, cols_normalize.tolist()