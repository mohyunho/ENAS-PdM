#!/bin/python3
"""
This file contains the implementation of a Task, used to load the data and compute the fitness of an individual
Author: Leonardo Lucio Custode
Date: 17/09/2020
"""
import pandas as pd
from abc import abstractmethod
from utils import network_launcher
# from .utils import network_input_generator, network_training


class Task:
    @abstractmethod
    def get_n_parameters(self):
        pass

    @abstractmethod
    def get_parameters_bounds(self):
        pass

    @abstractmethod
    def evaluate(self, genotype):
        pass


class SimpleNeuroEvolutionTask(Task):
    def __init__(self, dataframe_norm, cols_non_sensor, cols_sensors, model_path, fitness_mode, log_file_path, dropout, sequence_length=30, n_channel=1, strides_len=1, n_outputs=1, cross_val=True, k_value_fold=10, val_split=0.2, max_epoch=40, patience=10, bidirec=False, test_engine_idx=None, stride=1, piecewise_lin_ref=125, batch_size=400, experiment=False):
        self.dataframe_norm = dataframe_norm
        self.cols_non_sensor = cols_non_sensor
        self.cols_sensors = cols_sensors
        self.model_path = model_path
        self.fitness_mode = fitness_mode,
        self.log_file_path = log_file_path,
        self.dropout = dropout
        self.n_channel = n_channel
        self.strides_len = strides_len
        self.n_outputs = n_outputs
        self.cross_val = cross_val
        self.k_value_fold = k_value_fold
        self.val_split = val_split
        self.max_epoch = max_epoch
        self.patience = patience
        self.bidirec = bidirec
        self.test_engine_idx = test_engine_idx
        self.stride = stride
        self.piecewise_lin_ref = piecewise_lin_ref
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.experiment = experiment
        # self.n_window = n_window

    def get_n_parameters(self):
        return 5

    def get_parameters_bounds(self):
        bounds = [
            (1,5), # (3, self.sequence_length - 4),
            # (1, 8),  # (3, self.sequence_length - 4),

            (1, 10),

            # (1, int((self.sequence_length)/4)), # (3, int((self.sequence_length-4)/2)),

            (1, 2),

            # (8, 40), #(self.n_window, 50),
            # (8, 30), #(self.n_window, 50)
            (4, 20), #(self.n_window, 50),
            (4, 15), #(self.n_window, 50)
        ]
        return bounds

    def evaluate(self, genotype):
        # print ("genotype", genotype)
        # print ("len(genotype)", len(genotype))
        """ Creates a new instance of the training-validation task and computes the fitness of the current individual """
        training_input, training_input_label = network_launcher().network_input_generator(
            dataframe_norm=self.dataframe_norm,
            cols_non_sensor=self.cols_non_sensor,
            test_engine_idx=self.test_engine_idx,
            sequence_length=self.sequence_length,
            stride=self.stride,
            window_length=genotype[0],
            piecewise_lin_ref=self.piecewise_lin_ref
        )

        # print ("len(training_input)", len(training_input))


        fitness_net = network_launcher().network_training(
            training_input=training_input,
            training_input_label=training_input_label,
            cols_sensors=self.cols_sensors,
            model_path=self.model_path,
            fitness_mode=self.fitness_mode,
            log_file_path=self.log_file_path,
            dropout = self.dropout,
            test_engine_idx=self.test_engine_idx,
            n_channel=self.n_channel,
            n_filters=genotype[1],
            strides_len=self.strides_len,
            kernel_size=genotype[0], # if genotype[2] < genotype[0] else genotype[0],  # Saturation
            n_conv_layer=genotype[2],
            LSTM1_ref=genotype[3],
            # LSTM2_ref=genotype[4],
            LSTM2_ref=genotype[4] if genotype[4] < genotype[3] else genotype[3],  # Saturation
            n_outputs=self.n_outputs,
            cross_val=self.cross_val,
            k_value_fold=self.k_value_fold,
            val_split=self.val_split,
            batch_size=self.batch_size,
            max_epoch=self.max_epoch,
            patience=self.patience,
            bidirec=self.bidirec,
            experiment=self.experiment,
            geno_list = genotype
        )


        # aic = validation_accuracy + params_trainable_count


        return fitness_net

