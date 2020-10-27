from Classes.DataProcessing.LoadData import LoadData
from Classes.DataProcessing.BaselineHelperFunctions import BaselineHelperFunctions
from .GridSearchResultProcessor import GridSearchResultProcessor
import os
import sys
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join


class ResultFitter(GridSearchResultProcessor):
    
    def __init__(self, num_classes, is_balanced = True, shuffle = False):
        super().__init__()
        self.full_ds, self.train_ds, self.val_ds, self.test_ds = LoadData(num_classes = num_classes, 
                                                                          isBalanced = is_balanced, 
                                                                          shuffle = shuffle).getDatasets(shuffle = shuffle)
        self.helper = BaselineHelperFunctions()
    

    def fit_from_csv_and_index(self, file_name, index, num_classes, use_tensorboard = False, 
                               use_liveplots = False, use_custom_callbacks = False):

        # Major parameter parser
        df = GridSearchResultProcessor().get_results_df_by_name(file_name, num_classes)
        model_nr, detrend, use_scaler, use_minmax, use_noise_augmentor, use_early_stopping, use_highpass, highpass_freq = parse_result_name(file_name)

        if use_scaler:
            if use_minmax:
                scaler = MinMaxScalerFitter(self.train_ds).fit_scaler(test = False, detrend = detrend)
            else:
                scaler = StandardScalerFitter(self.train_ds).fit_scaler(test = False, detrend = detrend)
        else:
            scaler = None
        if use_noise_augmentor:
            augmentor = NoiseAugmentor(self.train_ds, use_scaler, scaler)
        else:
            augmentor = None

        values = list(df.iloc[index][0:13])
        keys = list(df.columns[0:13])
        params = {keys[i]: values[i] for i in range(len(keys))}

        build_model_args = self.helper.generate_build_model_args(model_nr, int(params['batch_size']), 
                                                                 float(params['dropout_rate']), params['activation'], 
                                                                 params['output_layer_activation'], float(params['l2_r']), 
                                                                 float(params['l1_r']), int(params['start_neurons']),
                                                                 int(params['filters']), int(params['kernel_size']),
                                                                 params['padding'], num_classes)
        # Build model using args generated above
        model = Models(**build_model_args).model

        # Generate generator args using picks.
        gen_args = self.helper.generate_gen_args(int(params['batch_size']), False, self.detrend, 
                                                 use_scaler = use_scaler, scaler = scaler, 
                                                 use_noise_augmentor = use_noise_augmentor, 
                                                 augmentor = augmentor, num_classes = num_classes)

        # Initiate generators using the args
        train_gen = self.data_gen.data_generator(self.train_ds, **gen_args)
        val_gen = self.data_gen.data_generator(self.val_ds, **gen_args)
        test_gen = self.data_gen.data_generator(self.test_ds, **gen_args)

        # Generate compiler args using picks
        opt = self.getOptimizer(params['optimizer'], float(params['learning_rate']))
        model_compile_args = self.helper.generate_model_compile_args(opt, num_classes)
        # Compile model using generated args
        model.compile(**model_compile_args)

        # Generate fit args using picks.
        fit_args = self.helper.generate_fit_args(self.train_ds, self.val_ds, int(params['batch_size']), False, 
                                                 int(params['epochs']), test_gen, use_tensorboard = use_tensorboard, 
                                                 use_liveplots = use_liveplots, 
                                                 use_custom_callback = use_custom_callback,
                                                 use_early_stopping = use_early_stopping)
        # Fit the model using the generated args
        model_fit = model.fit(train_gen, **fit_args)

        self.helper.plot_confusion_matrix(model, test_gen, self.test_ds, int(params['batch_size']), num_classes)

        # Evaluate the fitted model on the test set
        loss, accuracy, precision, recall = model.evaluate_generator(generator=test_gen,
                                                                   steps=self.helper.get_steps_per_epoch(self.test_ds, 
                                                                                                         int(params['batch_size']), 
                                                                                                         False))

        pp = pprint.PrettyPrinter(indent=4)
        print(f'Test loss: {loss}')
        print(f'Test accuracy: {accuracy}')
        print(f'Test precision: {precision}')
        print(f'Test recall: {recall}')
        return model    




    def parse_result_name(self, file_name):
        file_name = os.path.splitext(file_name)[0]
        major_params = file_name.split('_')[1:]
        model_nr = major_params[0]
        del major_params[0]

        use_scaler = 'sscale' in major_params
        use_noise_augmentor = 'noiseAug' in major_params
        detrend = 'detrend' in major_params
        use_minmax = 'mmscale' in major_params
        use_early_stopping = 'earlyS' in major_params
        use_highpass = False
        highpass_freq = 0.1
        for word in major_params:
            if len(word.split('-')) == 2:
                use_highpass = True
                highpass_freq = float(word.split('-')[1])

        return model_nr, detrend, use_scaler, use_minmax, use_noise_augmentor, use_early_stopping, use_highpass, highpass_freq