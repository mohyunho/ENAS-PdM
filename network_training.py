'''
Created on August 17, 2020
@author: hmo (hyunho.mo@unitn.it)
'''
import logging
import sys, os
import math
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
# import keras
import tensorflow as tf
print(tf.__version__)

# import keras.backend as K
import tensorflow.keras.backend as K
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding
from tensorflow.keras.layers import BatchNormalization, Activation, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

os.environ['TF_DETERMINISTIC_OPS'] = '1'




class network_train(object):
    '''
    class for network generation and training
    '''

    def __init__(self):
        '''
        Constructor
        @param none
        '''
        self.__logger = logging.getLogger("generating the multi-head CNN-LSTM network and"
                                          " training the network with prepared input")


    @staticmethod
    def set_gpu(gpu_ids_list):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                gpus_used = [gpus[i] for i in gpu_ids_list]
                tf.config.set_visible_devices(gpus_used, 'GPU')
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)


    @staticmethod
    def sensors_input_tensor(sensor_col, n_window, window_length, n_channel):
        '''
        Defining input tensors for multi-head architecture
        :param sensor_col: selected sensor columns in dataframe that are used as the network input
        :param : please refer to the description of arguments in utils.py
        :return: the shape of parallel branchs' input (python list)
        '''
        sensor_input_model = []
        for sensor in sensor_col:
            input_temp = Input(shape=(n_window, window_length, n_channel), name='%s' % sensor)
            sensor_input_model.append(input_temp)

        return sensor_input_model


    @staticmethod
    def TD_CNNBranch(n_filters, window_length, n_window, n_channel, strides_len, kernel_size, n_conv_layer):
        '''
        Defining time distributed cnn layers for each input branch of multi-head network
        :param : please refer to the description of arguments in utils.py
        :return: cnn architecture with following parameter settings.
        '''
        cnn = Sequential()
        cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same'),
                                input_shape=(n_window, window_length, n_channel)))
        cnn.add(TimeDistributed(BatchNormalization()))
        cnn.add(TimeDistributed(Activation('relu')))

        if n_conv_layer == 1:
            pass

        else:
            for loop in range(n_conv_layer-1):
                cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same')))
                cnn.add(TimeDistributed(BatchNormalization()))
                cnn.add(TimeDistributed(Activation('relu')))


        # elif n_conv_layer == 2:
        #     cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same')))
        #     cnn.add(TimeDistributed(BatchNormalization()))
        #     cnn.add(TimeDistributed(Activation('relu')))
        #
        # elif n_conv_layer == 3:
        #     cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same')))
        #     cnn.add(TimeDistributed(BatchNormalization()))
        #     cnn.add(TimeDistributed(Activation('relu')))
        #     cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same')))
        #     cnn.add(TimeDistributed(BatchNormalization()))
        #     cnn.add(TimeDistributed(Activation('relu')))

        cnn.add(TimeDistributed(Flatten()))
        # print(cnn.summary())

        return cnn


    @staticmethod
    def multi_head_cnn(sensor_input_model, n_filters, window_length, n_window,
                       n_channel, strides_len, kernel_size, n_conv_layer):
        '''
        Defining multi-head CNN network
        :param : please refer to the description of arguments in utils.py
        :return: multi-head CNN network  architecture with following parameter settings.
        '''

        cnn_out_list = []
        cnn_branch_list = []

        for sensor_input in sensor_input_model:
            cnn_branch_temp = network_train.TD_CNNBranch(n_filters, window_length, n_window,
                                                         n_channel, strides_len, kernel_size, n_conv_layer)
            cnn_out_temp = cnn_branch_temp(sensor_input)

            cnn_branch_list.append(cnn_branch_temp)
            cnn_out_list.append(cnn_out_temp)

        return cnn_out_list, cnn_branch_list

    @staticmethod
    def r2_keras(y_true, y_pred):
        """Metrics for evaluation"""
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))

    @staticmethod
    def rmse(y_true, y_pred):
        """Metrics for evaluation"""
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    # @staticmethod
    # def score(y_true, y_pred):
    #
    #     err = y_pred - y_true
    #     mask = K.less(y_pred, y_true)  # element-wise True where y_pred < y_pred
    #     mask = K.cast(mask, K.floatx())  # cast to 0.0 / 1.0
    #     score = mask * (K.exp(err / 10) - 1) + (1 - mask) * (K.exp(-err / 13) - 1)
    #     # every i where mask[i] is 1, s[i] == (K.exp(-d / 10) - 1)
    #     # every i where mask[i] is 0, s[i] == (K.exp(d / 13) - 1)
    #     return K.sum(score)

    @staticmethod
    def score(y_true, y_pred):
        y_error = y_pred - y_true
        bool_idx = K.greater(y_error, 0)
        loss1 = K.exp(-y_error / 13) - 1  # less 0
        loss2 = K.exp(y_error / 10) - 1  # greater 0
        loss = K.switch(bool_idx, loss2, loss1)
        return K.sum(loss)

        # h_array = y_predicted - y_actual
        # # print (h_array)
        # s_array = np.zeros(len(h_array))
        # for j, h_j in enumerate(h_array):
        #     if h_j < 0:
        #         s_array[j] = math.exp(-(h_j / 13)) - 1
        #
        #     else:
        #         s_array[j] = math.exp(h_j / 10) - 1
        #
        # score = np.sum(s_array)

    @staticmethod
    def shuffle_train_list_single(train_input, label_array_train):
        '''
        Shuffling training data without considering k-fold cross validation
        :param : please refer to the description of arguments in utils.py
        :return: shuffled training input and it's label
        '''
        train_input_shuffle = []

        # generate shuffle idx
        train_num_samples = label_array_train.shape[0]
        shuffle_idx = np.arange(train_num_samples)
        np.random.shuffle(shuffle_idx)

        # shuffle label
        label_array_train_shuffle = label_array_train[shuffle_idx]

        # Shuffle training samples per each sensor
        for s_idx in range(len(train_input)):
            temp_train_s_array = train_input[s_idx]
            temp_train_s_array_shuffle = temp_train_s_array[shuffle_idx]
            train_input_shuffle.append(temp_train_s_array_shuffle)

        return train_input_shuffle, label_array_train_shuffle



    @staticmethod
    def shuffle_train_list(train_input, label_array_train, k_value_fold):
        '''
        Shuffling training data before applying k-fold cross validation
        :param : please refer to the description of arguments in utils.py
        :return: shuffled training input and it's label
        '''
        train_input_shuffle = []
        # cut label of training instance
        temp_num_instances = label_array_train.shape[0]
        cut_residual = temp_num_instances % k_value_fold
        label_array_train_cut = label_array_train[cut_residual:, :]

        # generate shuffle idx
        train_num_samples = label_array_train_cut.shape[0]
        shuffle_idx = np.arange(train_num_samples)
        # np.random.shuffle(shuffle_idx)

        # shuffle label
        label_array_train_cut_shuffle = label_array_train_cut[shuffle_idx]

        for s_idx in range(len(train_input)):
            temp_train_s_array = train_input[s_idx]
            temp_train_s_array_cut = temp_train_s_array[cut_residual:, :, :, :]
            temp_train_s_array_cut_shuffle = temp_train_s_array_cut[shuffle_idx]
            train_input_shuffle.append(temp_train_s_array_cut_shuffle)

        return train_input_shuffle, label_array_train_cut_shuffle


    @staticmethod
    def cut_training_samples(train_input, label_array_train, val_split):
        temp_num_instances = label_array_train.shape[0]
        max_num_training_sample = int (temp_num_instances*(1-val_split))
        train_input_label_cut = label_array_train[:max_num_training_sample, :]
        train_num_samples = train_input_label_cut.shape[0]
        print ("num_training_samples: ", train_num_samples)

        train_input_cut = []

        for s_idx in range(len(train_input)):
            temp_train_s_array = train_input[s_idx]
            temp_train_s_array_cut = temp_train_s_array[:max_num_training_sample, :, :, :]
            train_input_cut.append(temp_train_s_array_cut)

        print ("num_training_samples: ", len(train_input_cut[0]))

        return train_input_cut, train_input_label_cut


    @staticmethod
    def split_train_sets(train_input_shuffle, label_array_train_cut_shuffle, k_value_fold):
        '''
        Splitting shuffled training input and it's label into k number of folds
        :param : please refer to the description of arguments in utils.py
        :return: the lists of folds (each fold has training and validation split)
        '''
        train_data_list_cross = []
        train_label_list_cross = []
        valid_data_list_cross = []
        valid_label_list_cross = []

        for k in range(k_value_fold):

            train_data_sensor_list = []
            valid_data_sensor_list = []
            label_array_split_list = np.split(label_array_train_cut_shuffle, k_value_fold)
            valid_label_array_k = label_array_split_list.pop(k)
            train_label_array_k = np.concatenate(label_array_split_list)

            for s_idx in range(len(train_input_shuffle)):
                combined_sensor_array_temp = train_input_shuffle[s_idx]
                combined_sensor_split_list = np.split(combined_sensor_array_temp, k_value_fold)
                valid_data_array_sensor = combined_sensor_split_list.pop(k)
                train_data_array_sensor = np.concatenate(combined_sensor_split_list)
                train_data_sensor_list.append(train_data_array_sensor)
                valid_data_sensor_list.append(valid_data_array_sensor)

            train_data_list_cross.append(train_data_sensor_list)
            valid_data_list_cross.append(valid_data_sensor_list)
            train_label_list_cross.append(train_label_array_k)
            valid_label_list_cross.append(valid_label_array_k)

        return train_data_list_cross, train_label_list_cross, valid_data_list_cross, valid_label_list_cross


    @staticmethod
    def scheduler(epoch, lr):
        if epoch == 10:
            return lr * 0.1
        elif epoch == 15:
            return lr * 0.1
        elif epoch == 20:
            return lr * tf.math.exp(-0.1)
        else:
            return lr


    def network_training(self, training_input, training_input_label, cols_sensors, model_path,
                         fitness_mode, log_file_path,
                         LSTM1_units, LSTM2_units, cross_val, k_value_fold, val_split, n_channel,
                         n_filters, strides_len, kernel_size, window_length, n_window,
                         n_conv_layer, n_outputs, batch_size, max_epoch, patience, bidirec, dropout, experiment, geno_list):
        '''
        :param : please refer to the description of arguments in utils.py
        :return: trained network is saved to model path
        '''
        ### Check the existence of trained model
        print ("Initializing network...")
        start_itr = time.time()
        # network_train.set_gpu([])
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:

                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    # tf.config.experimental.set_memory_growth(gpu, True)
                    tf.config.experimental.set_virtual_device_configuration(gpu, [
                        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5014)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        ### Training the network
        ## Use k-fold cross validation
        if cross_val == True:

            print(window_length, n_filters , kernel_size , n_conv_layer , LSTM1_units , LSTM2_units)
            print("n_window: ", n_window)

            model_cand_list = [] # list of k number of candidate model (the number of element is same as k)
            min_val_loss_list = [] # list of k number of validation loss of each candidate model
            min_val_rmse_list = [] # list of k number of validation RMSE of each candidate model
            # Shuffle and K split training set for cross validation
            train_input_shuffle, label_array_train_cut_shuffle = network_train.shuffle_train_list(training_input,
                                                                                    training_input_label, k_value_fold)
            train_data_list_cross, train_label_list_cross, valid_data_list_cross, valid_label_list_cross = \
            network_train.split_train_sets(train_input_shuffle, label_array_train_cut_shuffle, k_value_fold)

            ## Generate k models and training
            for k in range(k_value_fold):
                # Define the input tensor and multi-head CNN with static functions
                sensor_input_model = network_train.sensors_input_tensor(cols_sensors, n_window, window_length,
                                                                      n_channel)
                cnn_out_list, cnn_branch_list = network_train.multi_head_cnn(sensor_input_model, n_filters,
                                                                             window_length,
                                                                             n_window, n_channel, strides_len,
                                                                             kernel_size,
                                                                             n_conv_layer)

                # Concatenate the output of the each branches of multi-head CNN
                x = concatenate(cnn_out_list)

                # We stack a deep densely-connected network on top
                if bidirec == True:
                    x = Bidirectional(LSTM(units=LSTM1_units, return_sequences=True))(x)
                    x = Bidirectional(LSTM(units=LSTM2_units, return_sequences=False))(x)
                elif bidirec == False:
                    x = LSTM(units=LSTM1_units, return_sequences=True)(x)
                    x = LSTM(units=LSTM2_units, return_sequences=False)(x)

                # x = Dropout(0.5)(x)
                main_output = Dense(n_outputs, activation='linear', name='main_output')(x)

                cnnlstm_cand = Model(inputs=sensor_input_model, outputs=main_output)
                cnnlstm_cand.compile(loss='mean_squared_error', optimizer='rmsprop',
                                     metrics=['accuracy',network_train.rmse, 'mae', network_train.r2_keras, 'cosine_proximity'])

                print(cnnlstm_cand.summary())

                print("train model idx: ", k)
                # Train the network for each iteration
                history = cnnlstm_cand.fit(train_data_list_cross[k], train_label_list_cross[k], epochs=max_epoch,
                                           batch_size=batch_size,
                                           validation_data=(valid_data_list_cross[k], valid_label_list_cross[k]),
                                           verbose=2,
                                           callbacks=[EarlyStopping(monitor='val_loss', min_delta=0,
                                                                    patience=patience, verbose=1, mode='min')]
                                           )

                # Each trained model and it's validation loss&RMSE are saved to the lists, respectively
                val_loss_k = history.history['val_loss']
                val_rmse_k = history.history['val_rmse']

                model_cand_list.append(cnnlstm_cand)
                min_val_loss_list.append(min(val_loss_k))

                min_loss_idx = val_loss_k.index(min(val_loss_k))
                min_val_rmse_list.append(val_rmse_k[min_loss_idx])

            # Find the idx of best model which shows the lowest validation loss after training.
            best_model_idx = min_val_loss_list.index(min(min_val_loss_list))

            print("min_val_loss_list: ", min_val_loss_list)
            print("best_model_idx: ", best_model_idx)

            val_loss_min_tuple = (best_model_idx,)
            # Select the best model in the list of model candidates
            cnnlstm = model_cand_list[best_model_idx]
            # Save the best model
            cnnlstm.save(model_path)




        ## Training the network without using cross validation (preferable for cpu computation)
        elif cross_val == False:


            # Define the input tensor and multi-head CNN with static functions
            sensor_input_model = network_train.sensors_input_tensor(cols_sensors, n_window, window_length,
                                                                  n_channel)


            val_rmse_list = []
            stop_epoch_list = []



            np.random.seed(0)
            tf.random.set_seed(0)
            rp = optimizers.RMSprop(learning_rate=0.001, rho=0.9, centered=True)
            ##generate network
            cnn_out_list, cnn_branch_list = network_train.multi_head_cnn(sensor_input_model, n_filters,
                                                                         window_length,
                                                                         n_window, n_channel, strides_len,
                                                                         kernel_size,
                                                                         n_conv_layer)

            # Concatenate the output of the each branches of multi-head CNN
            x = concatenate(cnn_out_list)

            # We stack a deep densely-connected network on top
            if bidirec == True:
                x = Bidirectional(LSTM(units=LSTM1_units, return_sequences=True))(x)
                x = Bidirectional(LSTM(units=LSTM2_units, return_sequences=False))(x)
            elif bidirec == False:
                x = LSTM(units=LSTM1_units, return_sequences=True)(x)
                x = LSTM(units=LSTM2_units, return_sequences=False)(x)

            x = Dropout(0.5)(x)
            main_output = Dense(n_outputs, activation='linear', name='main_output')(x)
            cnnlstm = Model(inputs=sensor_input_model, outputs=main_output)

            # print(cnnlstm.summary())
            lr_scheduler = LearningRateScheduler(network_train.scheduler)
            keras_rmse = tf.keras.metrics.RootMeanSquaredError()

            cnnlstm.compile(loss='mean_squared_error', optimizer=rp,
                            metrics=[network_train.score, keras_rmse, 'mae'])

            # fit the network
            # history = cnnlstm.fit(training_input, training_input_label,
            #                       epochs=max_epoch, batch_size=batch_size, validation_split=val_split, verbose=1,
            #                       callbacks=[lr_scheduler,
            #                           EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0,
            #                                         patience=patience, verbose=1, mode='min')]
            #                       )

            history = cnnlstm.fit(training_input, training_input_label,
                                  epochs=max_epoch, batch_size=batch_size, validation_split=val_split, verbose=0,
                                  callbacks=[lr_scheduler,
                                      EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0,
                                                    patience=patience, verbose=1, mode='min')]
                                  )

            val_rmse_k = history.history['val_root_mean_squared_error']
            val_rmse_min = min(val_rmse_k)
            min_val_rmse_idx = val_rmse_k.index(min(val_rmse_k))
            # stop_epoch = min_val_rmse_idx +1

            val_score_k = history.history['val_score']
            val_score_min = min(val_score_k)
            min_val_score_idx = val_score_k.index(min(val_score_k))
            stop_epoch = min_val_score_idx +1


            # tracker
            print(window_length, n_filters , kernel_size , n_conv_layer , LSTM1_units , LSTM2_units)
            print("n_window: ", n_window)



            if experiment==False:

                ## Fitness aic
                params_trainable_count = np.sum([K.count_params(w) for w in cnnlstm.trainable_weights])
                # params_non_trainable_count = np.sum([K.count_params(w) for w in cnnlstm.non_trainable_weights])
                epoch = len(val_rmse_k)
                num_train_samples = int(len(training_input_label)*0.9)
                train_loss_k = history.history['loss']
                train_loss = train_loss_k[min_val_rmse_idx]
                # AIC = number of samples * log(training loss) + 2 * number of parameters
                mle_term =  num_train_samples * np.log(train_loss)
                params_term = 2*params_trainable_count
                aic = mle_term + params_term
                print("AIC: ", aic)

                # combined = rmse + aic * 1e-7
                # combined = score + aic * 1e-5

                rmse_combined = val_rmse_min + aic * 1e-7
                score_combined = val_score_min + aic * 1e-5

                temp_df = pd.DataFrame(np.array([[np.float64(stop_epoch), np.float64(window_length), np.float64(n_filters) ,
                                                  np.float64(kernel_size) , np.float64(n_conv_layer) , np.float64(LSTM1_units) , np.float64(LSTM2_units), np.float64(n_window),
                                                  np.float64(val_rmse_min), np.float64(val_score_min), np.float64(rmse_combined), np.float64(score_combined),
                                                  np.float64(aic), np.float64(train_loss), np.float64(mle_term), np.float64(params_term), geno_list]]))


                temp_df = temp_df.round(4)
                temp_df.to_csv(log_file_path[0], mode='a', header=None)


                val_rmse_min = round(val_rmse_min,4)
                val_score_min = round(val_score_min, 4)

                rmse_combined = round(rmse_combined,4)
                score_combined = round(score_combined, 4)

                aic = round(aic,4)

                if fitness_mode[0]==0:
                    fitness_net = (val_rmse_min,)
                elif fitness_mode[0]==1:
                    fitness_net = (aic,)
                elif fitness_mode[0]==2:
                    fitness_net = (val_score_min,)
                elif fitness_mode[0]==3:
                    fitness_net = (rmse_combined,)
                elif fitness_mode[0] == 4:
                    fitness_net = (score_combined,)

                print("val_score_min", val_score_min)
                print("val_rmse_min", val_rmse_min)
                print ("fitness_net", fitness_net[0])

            else:
                print ("test val_rmse_min: ", val_rmse_min)
                fitness_net = cnnlstm


        end_itr = time.time()


        print("training network is successfully completed, time: ", end_itr - start_itr)
        return fitness_net




    def network_test(self, training_input, training_input_label, test_engine_idx, model_path):
        '''
        :param : please refer to the description of arguments in utils.py
        :return: test(apply) the trained model and show the performance
        :y_predicted: predicted RUL of the selected engine (python list)
        :y_actual: ground truth RUL of the selected engine (python list)
        '''
        ### Check the existence of trained model
        try:
            estimator = load_model(model_path, custom_objects={'rmse': network_train.rmse, 'r2_keras': network_train.r2_keras})
            # test metrics
            scores_test = estimator.evaluate(training_input, training_input_label, verbose=2)

            y_pred_test = estimator.predict(training_input)
            y_true_test = training_input_label

            pd.set_option('display.max_rows', 1000)
            test_print = pd.DataFrame()
            test_print['y_pred'] = y_pred_test.flatten()
            test_print['y_truth'] = y_true_test.flatten()
            # print (test_print)

            y_predicted = test_print['y_pred']
            y_actual = test_print['y_truth']
            rms = sqrt(mean_squared_error(y_actual, y_predicted))
            test_print['rmse'] = rms

            h_array = y_predicted - y_actual
            # print (h_array)
            s_array = np.zeros(len(h_array))
            for j, h_j in enumerate(h_array):
                if h_j < 0:
                    s_array[j] = math.exp(-(h_j / 13)) - 1
                else:
                    s_array[j] = math.exp(h_j / 10) - 1

            score = np.sum(s_array)
            test_set = pd.DataFrame(y_pred_test)

            # Plot in blue color as  the predicted data and  green color as ground truth
            # actual data to verify visually the accuracy of the model.
            fig_rul = plt.figure(figsize=(12, 11))
            plt.plot(y_pred_test, color="blue")
            plt.plot(y_true_test, '--', color="green")

            plt.title('RUL Engine#: %s\n (RMSE: %.2f, score: %.2f)' % (test_engine_idx + 1, rms, score),
                      fontdict={'fontsize': 44})
            plt.ylabel('RUL(cycles)', fontsize=44)
            plt.xlabel('Cycle number', fontsize=44)
            plt.legend(['Predicted RUL', 'Ground truth RUL'], loc='lower left', fontsize=44)
            plt.ylim(0, 140)
            plt.xticks(fontsize=26)
            plt.yticks(fontsize=26)
            plt.show()
            fig_rul.savefig(
                '../tmp/' + "predicted_rul_graph_%s_engine_with_data.png" %str(test_engine_idx + 1))


        except IOError:
            print("Trained model is not exist")

        return y_predicted, y_actual



    def network_test_application(self, current_input, model_path):
        '''
        :param : please refer to the description of arguments in utils.py
        :return: test(apply) the trained model per for single instance
        '''
        ### Check the existence of trained model
        try:
            estimator = load_model(model_path, custom_objects={'rmse': network_train.rmse, 'r2_keras': network_train.r2_keras})
            y_pred_test = estimator.predict(current_input)

        except IOError:
            print("Trained model is not exist")

        return y_pred_test


    def opt_network_test_rmse(self, cnnlstm, test_FD_sensor, label_array_test_last, model_path, window_length,
                              n_filters ,kernel_size ,n_conv_layer ,LSTM1_ref ,LSTM2_ref):
        '''
        :param :
        :return:
        '''

        # estimator = load_model(model_path, custom_objects={'rmse': network_train.rmse, 'r2_keras': network_train.r2_keras})

        # # test metrics
        # scores_test = cnnlstm.evaluate(test_FD_sensor, label_array_test_last, verbose=2)
        # print("estimator.metrics_names", estimator.metrics_names)
        # print("scores_test", scores_test)
        # print('\nLOSS: {}'.format(scores_test[0]))
        # print('\nRMSE: {}'.format(scores_test[1]))
        # print('\nMAE: {}'.format(scores_test[2]))
        # print('\nR2_keras: {}'.format(scores_test[3]))

        y_pred_test = cnnlstm.predict(test_FD_sensor)
        y_true_test = label_array_test_last

        pd.set_option('display.max_rows', 1000)
        test_print = pd.DataFrame()
        test_print['y_pred'] = y_pred_test.flatten()
        test_print['y_truth'] = y_true_test.flatten()
        test_print['diff'] = abs(y_pred_test.flatten() - y_true_test.flatten())
        test_print['diff(ratio)'] = abs(y_pred_test.flatten() - y_true_test.flatten()) / y_true_test.flatten()
        test_print['diff(%)'] = (abs(y_pred_test.flatten() - y_true_test.flatten()) / y_true_test.flatten()) * 100
        # print(test_print)

        y_predicted = test_print['y_pred']
        y_actual = test_print['y_truth']
        rms = sqrt(mean_squared_error(y_actual, y_predicted))
        test_print['rmse'] = rms

        h_array = y_predicted - y_actual
        # print (h_array)
        s_array = np.zeros(len(h_array))
        for j, h_j in enumerate(h_array):
            if h_j < 0:
                s_array[j] = math.exp(-(h_j / 13)) - 1

            else:
                s_array[j] = math.exp(h_j / 10) - 1

        score = np.sum(s_array)

        print("RMSE: ", rms)
        print("Score: ", score)


        # Plot in blue color the predicted data and in green color the
        # actual data to verify visually the accuracy of the model.
        fig_verify = plt.figure(figsize=(12, 6))
        plt.plot(y_pred_test, color="blue")
        plt.plot(y_true_test, color="green")
        plt.title('prediction_RMSE-%.2f, score-%.2f' %(rms, score))
        plt.ylabel('value')
        plt.xlabel('row')
        plt.legend(['predicted', 'actual data'], loc='upper left')
        plt.show()
        # fig_verify.savefig(colab_nb_path +"/CIM/graph/predicted_final_rul_%s,%s,%s,%s,%s,%s.png" %(window_length, n_filters ,kernel_size ,n_conv_layer ,LSTM1_ref ,LSTM2_ref))






        return rms, score