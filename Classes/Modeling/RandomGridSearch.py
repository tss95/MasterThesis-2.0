import numpy as np
import pandas as pd
import h5py

import sklearn as sk
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ModelCheckpoint

import keras
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from tensorflow.keras import regularizers
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import ParameterGrid




import tensorflow as tf

from .Models import Models

from Classes.DataProcessing.LoadData import LoadData
from Classes.DataProcessing.BaselineHelperFunctions import BaselineHelperFunctions
from Classes.DataProcessing.DataGenerator import DataGenerator
from Classes.DataProcessing.NoiseAugmentor import NoiseAugmentor
from Classes.Scaling.MinMaxScalerFitter import MinMaxScalerFitter
from Classes.Scaling.StandardScalerFitter import StandardScalerFitter

import sys
import os


from livelossplot import PlotLossesKeras
import random
import pprint
import re
import json

base_dir = 'C:\Documents\Thesis_ssd\MasterThesis-2.0'

"""
Best so far:
{'batch_size': 16, 'epochs': 95, 'learning_rate': 0.1, 'optimizer': 'adam'},
 {'activation': 'tanh',
  'dropout_rate': 0.5,
  'l1_r': 0.1,
  'l2_r': 0.0001,
  'start_neurons': 128},
 {'loss': 358.6396789550781,
  'accuracy': 0.51602566242218018,
  'mse': 0.21034729480743408,
  'precision': 0.33800473809242249,
  'recall': 0.22719238698482513},
 {'train_loss': 358.9141845703125,
  'train_accuracy': 0.37990197539329529,
  'train_mse': 0.23922176659107208,
  'train_precision': 0.33883824944496155,
  'train_recall': 0.22711670398712158}
  
  Best training results:
  {'batch_size': 128,
  'epochs': 100,
  'learning_rate': 0.0001,
  'optimizer': 'rmsprop'},
 {'activation': 'tanh',
  'dropout_rate': 0.2,
  'l1_r': 0.2,
  'l2_r': 0.001,
  'start_neurons': 512},
 {'loss': 7.242753028869629,
  'accuracy': 0.3671875,
  'mse': 0.31585189700126648,
  'precision': 0.91042166948318481,
  'recall': 0.87738406658172607},
 {'train_loss': 4.252453804016113,
  'train_accuracy': 1.0,
  'train_mse': 0.00082400580868124962,
  'train_precision': 0.90749055147171021,
  'train_recall': 0.87465983629226685}
  """

#TODO: Implement highpass filter into this class.
#TODO: CLEAN UP THIS MESS
class RandomGridSearch():
    hyper_grid = {
            "batch_size" : [8, 16, 32, 64, 128, 256],
            "epochs" : [50, 65, 70, 75, 80],
            "learning_rate" : [0.1, 0.01, 0.001, 0.0001, 0.00001],
            "optimizer" : ["adam", "rmsprop", "sgd"]
        }
    model_grid = {
        "start_neurons" : [2,4,8,16, 32, 64, 128, 256, 512],
        "dropout_rate" : [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0],
        "filters" : [3, 9, 15, 21],
        "kernel_size" : [3],
        "padding" : ["same", "valid"],
        "l2_r" : [0.3, 0.2, 0.1, 0.01, 0.001, 0.0001],
        "l1_r" : [0.3, 0.2, 0.1, 0.01, 0.001, 0.0001],
        "activation" : ["relu", "sigmoid", "softmax", "tanh"],
        "output_layer_activation" : ["relu", "sigmoid", "softmax", "tanh"]
    }
    

    def __init__(self, train_ds, val_ds, test_ds, model_nr, test, detrend, use_scaler, use_noise_augmentor,
                 use_minmax, use_highpass, n_picks, hyper_grid=hyper_grid, model_grid=model_grid, num_classes = 3, 
                 use_tensorboard = False, use_liveplots = True, use_custom_callback = False, use_early_stopping = False,
                 highpass_freq = 0.1):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.model_nr = model_nr
        self.test = test
        self.detrend = detrend
        self.use_scaler = use_scaler
        self.use_noise_augmentor = use_noise_augmentor
        self.use_minmax = use_minmax
        self.use_highpass = use_highpass
        self.n_picks = n_picks
        self.hyper_grid = hyper_grid
        self.model_grid = model_grid
        self.num_classes = num_classes
        self.use_tensorboard = use_tensorboard
        self.use_liveplots = use_liveplots
        self.use_custom_callback = use_custom_callback
        self.use_early_stopping = use_early_stopping
        self.highpass_freq = highpass_freq
        self.helper = BaselineHelperFunctions()
        self.data_gen = DataGenerator()
    
    def create_results_df(self):
        hyper_keys = list(self.hyper_grid.keys())
        model_keys = list(self.model_grid.keys())
        metrics_train_keys = ["train_loss", "train_accuracy", "train_precision", "train_recall"]
        metrics_val_keys = ["val_loss", "val_accuracy", "val_precision", "val_recall"]
        header = np.concatenate((hyper_keys, model_keys, metrics_train_keys, metrics_val_keys))
        results_df = pd.DataFrame(columns = header)
        return results_df
    
    def save_results_df(self, results_df, file_name):
        results_df.to_csv(file_name, mode = 'a')
    
    def clear_results_df(self, file_name):
        path = self.get_results_file_path()
        file = f"{path}/{file_name}"
        if os.path.isfile(file):
            f = open(file, "w+")
            f.close()
        
    
    def get_results_file_name(self):
        file_name = f"{self.get_results_file_path}/results_{self.model_nr}"
        if self.test:
            file_name = f"{file_name}_test"
        if self.detrend:
            file_name = f"{file_name}_detrend"
        if self.use_scaler:
            if self.use_minmax:
                file_name = f"{file_name}_mmscale"
            else: 
                file_name = f"{file_name}_sscale"
        if self.use_noise_augmentor:
            file_name = f"{file_name}_noiseAug"
        if self.use_early_stopping:
            file_name = f"{file_name}_earlyS"
        if self.use_highpass:
            file_name = f"{file_name}_highpass{self.highpass_freq}"
        return file_name
    
    def get_results_file_path(self):
        file_path = f'{base_dir}/GridSearchResults/{self.num_classes}_classes'
    
    def store_params_before_fit(self, current_picks, results_df, file_name):
        hyper_params = current_picks[1]
        model_params = current_picks[2]
        picks = []
        for key in list(hyper_params.keys()):
            picks.append(hyper_params[key])
        for key in list(model_params.keys()):
            picks.append(model_params[key])
        nr_fillers = len(results_df.columns) - len(picks)
        for i in range(nr_fillers):
            picks.append(np.nan)
        temp_df = pd.DataFrame(np.array(picks).reshape(1,21), columns = results_df.columns)
        results_df.append(temp_df)
        self.save_results_df(results_df, file_name)
        return results_df


    def store_metrics_after_fit(self, metrics, results_df, file_name):
        metrics_train = metrics[0]
        metrics_val = metrics[1]
        finished_train = False
        # Get last row in results
        temp_df = results_df.tail(1)
        # Get list of columns containing nan values
        unfinished_columns = temp_df.columns[temp_df.isnull().any()].tolist()
        # Iterate through every unfinished column and change values
        for idx, column in enumerate(unfinished_columns):
            if not finished_train:
                if idx >= len(metrics_train):
                    finished_train = True
                # Change value at column using the metrics
                results_df.iloc[-1, a.columns.get_loc(column)] = metrics_train[column]
            
            else:
                # Change value at column using the metrics
                results_df.iloc[-1, a.columns.get_loc(column)] = metrics_val[column]
        self.save_results_df(results_df, file_name)
        return results_df
            

    def fit(self):
        if self.use_scaler:
            if self.use_minmax:
                self.scaler = MinMaxScalerFitter(self.train_ds).fit_scaler(test = self.test, detrend = self.detrend)
            else:
                self.scaler = StandardScalerFitter(self.train_ds).fit_scaler(test = self.test, detrend = self.detrend)
        else:
            self.scaler = None
        if self.use_noise_augmentor:
            self.augmentor = NoiseAugmentor(self.train_ds, self.use_scaler, self.scaler)
        
        # Create name of results file, clear contents if one exists, create df to work with.
        self.results_file_name = self.get_results_file_name()
        self.clear_results_df(self.results_file_name)
        self.results_df = self.create_results_df()
        
        self.hyper_picks = self.get_n_params_from_list(list(ParameterGrid(self.hyper_grid)), self.n_picks)
        self.model_picks = self.get_n_params_from_list(list(ParameterGrid(self.model_grid)), self.n_picks)
        self.results = []
        pp = pprint.PrettyPrinter(indent=4)
        for i in range(self.n_picks):
            model_info = {"model_nr" : self.model_nr, "index" : i}
            current_picks = [model_info, self.hyper_picks[i], self.model_picks[i]]
            # Store picked parameters:
            self.results_df = self.store_params_before_fit(current_picks, self.results_df, self.results_file_name)
            
            # Translate picks to a more readable format:
            epoch = self.hyper_picks[i]["epochs"]
            batch_size = self.hyper_picks[i]["batch_size"]
            dropout_rate = self.model_picks[i]["dropout_rate"]
            activation = self.model_picks[i]["activation"]
            output_layer_activation = self.model_picks[i]["output_layer_activation"]
            l2_r = self.model_picks[i]["l2_r"]
            l1_r = self.model_picks[i]["l1_r"]
            start_neurons = self.model_picks[i]["start_neurons"]
            filters = self.model_picks[i]["filters"]
            kernel_size = self.model_picks[i]["kernel_size"]
            padding = self.model_picks[i]["padding"]
            opt = self.getOptimizer(self.hyper_picks[i]["optimizer"], self.hyper_picks[i]["learning_rate"])
            
            # Generate build model args using the picks from above.
            build_model_args = self.helper.generate_build_model_args(self.model_nr, batch_size, dropout_rate, 
                                                                     activation, output_layer_activation,
                                                                     l2_r, l1_r, start_neurons, filters, kernel_size, 
                                                                     padding, self.num_classes)
            # Build model using args generated above
            model = Models(**build_model_args).model
            
            # Generate generator args using picks.
            gen_args = self.helper.generate_gen_args(batch_size, self.test, self.detrend, use_scaler = self.use_scaler, scaler = self.scaler, use_noise_augmentor = self.use_noise_augmentor, augmentor = self.augmentor, num_classes = self.num_classes)
            
            # Initiate generators using the args
            train_gen = self.data_gen.data_generator(self.train_ds, **gen_args)
            val_gen = self.data_gen.data_generator(self.val_ds, **gen_args)
            test_gen = self.data_gen.data_generator(self.test_ds, **gen_args)
            
            # Generate compiler args using picks
            model_compile_args = self.helper.generate_model_compile_args(opt, self.num_classes)
            # Compile model using generated args
            model.compile(**model_compile_args)
            
            print("Starting: ")
            pp.pprint(self.hyper_picks[i])
            print("---------------------------------------------------------------------------------")
            pp.pprint(self.model_picks[i])

            
            # Generate fit args using picks.
            fit_args = self.helper.generate_fit_args(self.train_ds, self.val_ds, batch_size, self.test, 
                                                     epoch, val_gen, use_tensorboard = self.use_tensorboard, 
                                                     use_liveplots = self.use_liveplots, 
                                                     use_custom_callback = self.use_custom_callback,
                                                     use_early_stopping = self.use_early_stopping)
            # Fit the model using the generated args
            model_fit = model.fit(train_gen, **fit_args)
            
            # Evaluate the fitted model on the validation set
            loss, accuracy, precision, recall = model.evaluate_generator(generator=val_gen,
                                                                         steps=self.helper.get_steps_per_epoch(self.val_ds, 
                                                                                                               batch_size, False))
            # Record metrics for train
            metrics = []
            metrics_val = {"val_loss" : loss,
                            "val_accuracy" : accuracy,
                            "val_precision": precision,
                            "val_recall" : recall}
            metrics.append(metrics_val)
            current_picks.append(metrics_val)
            
            # Evaluate the fitted model on the train set
            train_loss, train_accuracy, train_precision, train_recall = model.evaluate_generator(generator=train_gen,
                                                                                        steps=self.helper.get_steps_per_epoch(self.train_ds,
                                                                                                                              batch_size,
                                                                                                                              True))
            metrics_train = {"train_loss" : train_loss,
                             "train_accuracy" : train_accuracy,
                             "train_precision": train_precision,
                             "train_recall" : train_recall}
            metrics.append(metrics_train)
            current_picks.append(metrics_train)
            self.results_df = self.store_metrics_after_fit(metrics, self.results_df, self.results_file_name)
            self.results.append(current_picks)
            
        highest_test_accuracy_index, highest_train_accuracy_index, highest_test_precision_index, highest_test_recall_index = self.find_best_performers(self.results)
        return self.results, highest_test_accuracy_index, highest_train_accuracy_index, highest_test_precision_index, highest_test_recall_index

    def find_best_performers(self, results):
        highest_test_accuracy = 0
        highest_test_accuracy_index = 0
        highest_train_accuracy = 0
        highest_train_accuracy_index = 0
        highest_test_precision = 0
        highest_test_precision_index = 0
        highest_train_precision = 0
        highest_train_precision_index = 0
        highest_test_recall = 0
        highest_test_recall_index = 0
        highest_train_recall = 0
        highest_train_recall_index = 0
        for idx, result in enumerate(results):
            if result[3]["test_accuracy"] > highest_test_accuracy:
                highest_test_accuracy = result[3]["test_accuracy"]
                highest_test_accuracy_index = idx
            if result[4]["train_accuracy"] > highest_train_accuracy:
                highest_train_accuracy = result[4]["train_accuracy"]
                highest_train_accuracy_index = idx
            if result[3]["test_precision"] > highest_test_precision:
                highest_test_precision = result[3]["test_precision"]
                highest_test_precision_index = idx
            if result[4]["train_precision"] > highest_train_precision:
                highest_train_precision = result[4]["train_precision"]
                highest_train_precision_index = idx
            if result[3]["test_recall"] > highest_test_recall:
                highest_test_recall = result[3]["test_recall"]
                highest_test_recall_index = idx
            if result[4]["train_recall"] > highest_train_recall:
                highest_train_recall = result[4]["train_recall"]
                highest_train_recall_index = idx
        print("----------------------------------------------------ACCURACY------------------------------------------------------")
        print(f'Highest test accuracy: {highest_test_accuracy}, at index: {highest_test_accuracy_index}')
        print(f'Highest training accuracy: {highest_train_accuracy}, at index: {highest_train_accuracy_index}')
        print("----------------------------------------------------PRECISION-----------------------------------------------------")
        print(f'Highest test precision: {highest_test_precision}, at index: {highest_test_precision_index}')
        print(f'Highest training precision: {highest_train_precision}, at index: {highest_train_precision_index}') 
        print("-----------------------------------------------------RECALL-------------------------------------------------------")
        print(f'Highest test recall: {highest_test_recall}, at index: {highest_test_recall_index}')
        print(f'Highest training recall: {highest_train_recall}, at index: {highest_train_recall_index}')
        print("------------------------------------------------------------------------------------------------------------------")
        return highest_test_accuracy_index, highest_train_accuracy_index, highest_test_precision_index, highest_test_recall_index
       
    
    def read_results(self):
        text_file = f'{base_dir}/GridSearchResults/{self.num_classes}_classes/results_{self.model_nr}.txt'
        dictionaries = []
        with open(text_file, 'r') as file:
            for idx, line in enumerate(file):
                if idx == 0 or idx == 1:
                    continue
                line = re.sub("\'", "\"", line.rstrip())
                if idx + 2 % 5 != 0: 
                    print(line)
                    dictionaries.append(json.loads(line))
            file.close()
        dictionaries_by_model = []
        one_model = []
        for idx, dictionary in enumerate(dictionaries):
            if idx % 5 != 0 or idx == 0:
                one_model.append(dictionary)
            else:
                dictionaries_by_model.append(one_model)
                one_model = []
                one_model.append(dictionary)           
        return dictionaries_by_model
    
    def fit_from_result(self, dictionaries, index, train_channels = 3, timesteps = 6001, test = False, use_tensorboard = False):
        
        build_model_args = self.helper.generate_build_model_args(dictionaries[index][0]['model_nr'], dictionaries[index][1]['batch_size'], 
                                                     dictionaries[index][2]['dropout_rate'], dictionaries[index][2]['activation'], 
                                                     dictionaries[index][2]['l2_r'], dictionaries[index][2]['l1_r'],
                                                     dictionaries[index][2]['start_neurons'], dictionaries[index][2]['filters'],
                                                     dictionaries[index][2]['kernel_size'], dictionaries[index][2]['padding'])
        model = Models(**build_model_args).model

        model_compile_args = self.helper.generate_model_compile_args(dictionaries[index][1]['optimizer'])
        model.compile(**model_compile_args)
       
        gen_args = self.helper.generate_gen_args(dictionaries[index][1]['batch_size'], False, self.detrend, self.num_classes)
        
        train_gen = self.data_gen.data_generator(self.train_ds, **gen_args)
        val_gen = self.data_gen.data_generator(self.val_ds, **gen_args)
        test_gen = self.data_gen.data_generator(self.test_ds, **gen_args)

        fit_args = self.helper.generate_fit_args(self.train_ds, self.val_ds, dictionaries[index][1]['batch_size'], False, dictionaries[index][1]['epochs'], val_gen, use_tensorboard)
        model.fit(train_gen, **fit_args)
        return model
    
    def get_n_params_from_list(self, grid_list, n_picks):
        picks = []
        while (n_picks != 0):
            grid_length = len(grid_list)
            rand_int = random.randint(0,grid_length-1)
            picks.append(grid_list[rand_int])
            del grid_list[rand_int]
            n_picks -= 1
        return picks

    def getOptimizer(self, optimizer, learning_rate):
        if optimizer == "adam":
            return keras.optimizers.Adam(learning_rate=learning_rate)
        if optimizer == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        if optimizer == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise Exception(f"{optimizer} not implemented into getOptimizer")
    
    def save_conditionals(self):
        with open(f'{base_dir}/GridSearchResults/{self.num_classes}_classes/results_{self.model_nr}.txt', 'a', newline='') as file:
            file.writelines(f'Test_mode: {self.test}, use_scaler: {self.use_scaler}, use_minmax: {self.use_minmax}, use_noise_augmentor: {self.use_noise_augmentor} detrend: {self.detrend}. \n')
            file.close()
    
    def save_params(self, current_params):
        with open(f'{base_dir}/GridSearchResults/{self.num_classes}_classes/results_{self.model_nr}.txt', 'a', newline='') as file:
            file.writelines("-------------------------------------------------------------------------------------------------------------------------------------------\n")
            for part in current_params:
                file.writelines(str(part) + '\n')
            file.close()
            
    def save_metrics(self, metrics):
        with open(f'{base_dir}/GridSearchResults/{self.num_classes}_classes/results_{self.model_nr}.txt', 'a', newline='') as file:
            for part in metrics:
                file.writelines(str(part) + '\n')
            file.close()
    
    def delete_progress(self):
        with open(f'{base_dir}/GridSearchResults/{self.num_classes}_classes/results_{self.model_nr}.txt', 'w+', newline='') as file:
            file.truncate(0)

    
    
    
    
    
    
    
    
    
    
    
    