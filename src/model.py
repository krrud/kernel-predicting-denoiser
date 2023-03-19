from keras.layers import Input, Conv2D, Concatenate
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.optimizers import Adam
from .util import *
import datetime
import os
import re 
import keras.backend as K


class KPNModel:        
    """ Kernel predicting convolutional neural network for denoising renders """

    def __init__(self, kernel_size=21, rgb_shape=(64, 64, 3), feature_shape=(64, 64, 31)):
        self.kernel_size = kernel_size
        self.rgb_shape = rgb_shape
        self.feature_shape = feature_shape
        self.model = self.build_model()


    def build_model(self):

        # Inputs
        rgb_inputs = Input(self.rgb_shape)
        feature_inputs = Input(self.feature_shape)

        # RGB processing
        x = Conv2D(filters=100, kernel_size=5, padding='same', activation='relu')(rgb_inputs)
        x = Conv2D(filters=100, kernel_size=5, padding='same', activation='relu')(x)
        x = Conv2D(filters=100, kernel_size=5, padding='same', activation='relu')(x)
        x = Conv2D(filters=100, kernel_size=5, padding='same', activation='relu')(x)

        # Feature processing
        y = Conv2D(filters=100, kernel_size=5, padding='same', activation='relu')(feature_inputs)
        y = Conv2D(filters=100, kernel_size=5, padding='same', activation='relu')(y)
        y = Conv2D(filters=100, kernel_size=5, padding='same', activation='relu')(y)
        y = Conv2D(filters=100, kernel_size=5, padding='same', activation='relu')(y)

        # Concatenate rgb and features
        concat = Concatenate()([x, y])

        # Kernel prediction
        outputs = Conv2D(filters=3, kernel_size=self.kernel_size, padding='same', activation='relu')(concat)
        model = Model(inputs=[rgb_inputs, feature_inputs], outputs=outputs)

        return model


    def train(
            self,
            channel: str, 
            directory: str, 
            samples: int, 
            learning_rate: float,
            epochs: int,
            batch_size: int,
            weights_init: str,
            weights_save: str,
            load_train: bool=False,
            load_val: bool=False,
            augment: bool=False,
            cache: bool=False,
            ):
        """ Trains the network on exr's -which need to be preprocessed-
            or on cached .npz files which alleviates the need for preprocessing.
            
            channel: The channel to denoise - either diffuse or specular
            directory: Diretory of exr's to train on - ignored if train_load is True
            samples: How many 64x64 samples to take from the dataset - ignored if train_load is True
            learning_rate: Learning rate for the fit process 
            epochs: How many epochs for the fit process
            batch_size: Batch size for the fit process
            weights_init: Path to a .h5 file to initialize the model's weights - If None - the weights will randomly initialize
            weights_save: Path to .h5 output file
            load_train: If True the model will load a random .npz file from the data/train/npz_<channel>/ directory
            load_val: If True the model will load validation data from the data/val/npz_<channel>/
            augment: If True each sample will be randomly rotated to add artificial variation to the dataset 
            cache: If True the samples will be saved to disk as an .npz file """
        
        c = None
        if channel in ['diffuse', 'diff', 'd']:
            c = 'diffuse'
        elif channel in ['specular', 'spec', 's']:
            c = 'specular'
        else:
            raise ValueError(f"Invalid channel input: {c}")
        
        if load_train:
            # Randomly select cached .npz file to load
            npz_files = [file for file in os.listdir(f'data/train/npz_{c}/') if file.endswith('.npz')]
            sample_npz = random.sample(npz_files, 1)[0]
            sample_npz = f'data/train/npz_{c}/' + sample_npz
            print(f'\rLoading training data: {sample_npz}')

            # Load .npz file and extract variables
            train_data = np.load(sample_npz)
            x_train_rgb = train_data['x_train_rgb']
            x_train_feature = train_data['x_train_feature']
            y_train_rgb = train_data['y_train_rgb']
            y_train_feature = train_data['y_train_feature']

        else:
            # Randomly sample and preprocess the exr dataset
            train_data = data_prep(directory, channel, samples=samples, augment=augment)
            x_train_rgb, x_train_feature = train_data[0], train_data[1]
            y_train_rgb, y_train_feature = train_data[2], train_data[3]

            if cache:
                existing = ''.join([file for file in os.listdir(f'data/train/npz_{c}/') if file.endswith('.npz')])
                numbers = re.findall(r'\d+', existing)
                i = max([int(n) for n in numbers]) + 1
                np.savez(f'data/train/npz_{c}/train_{i}.npz',
                        x_train_rgb=x_train_rgb, x_train_feature=x_train_feature, 
                        y_train_rgb=y_train_rgb, y_train_feature=y_train_feature)

        if load_val:
            # Randomly select cached .npz file to load
            npz_files = [file for file in os.listdir(f'data/val/npz_{c}/') if file.endswith('.npz')]
            sample_npz = random.sample(npz_files, 1)[0]
            sample_npz = f'data/val/npz_{c}/' + sample_npz

            # Load .npz file and extract variables
            print(f'\rLoading validation data: {sample_npz}')
            val_data = np.load(sample_npz)
            x_val_rgb = val_data['x_val_rgb']
            x_val_feature = val_data['x_val_feature']
            y_val_rgb = val_data['y_val_rgb']
            y_val_feature = val_data['y_val_feature']

        else:
            # Randomly sample and preprocess the exr dataset
            count = 1000
            val_data = data_prep('data/val/exr/', channel, samples=count, augment=False)
            x_val_rgb, x_val_feature = val_data[0], val_data[1]
            y_val_rgb, y_val_feature = val_data[2], val_data[3]

            if cache:
                existing = ''.join([file for file in os.listdir(f'data/val/npz_{c}/') if file.endswith('.npz')])
                numbers = re.findall(r'\d+', existing)
                i = max([int(n) for n in numbers]) + 1
                np.savez(f'data/val/npz_{c}/val_{i}.npz',
                        x_val_rgb=x_val_rgb, x_val_feature=x_train_feature, 
                        y_val_rgb=y_val_rgb, y_val_feature=y_val_feature)

        # Instantiate training model and initialize weights 
        training_model = self.model
        if weights_init:
            pad = ' ' * 35
            print(f'\rLoading weights: {weights_init}' + pad)
            training_model.load_weights(weights_init)

        # Compile and fit
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        opt = Adam(learning_rate=learning_rate)
        mcp_save = ModelCheckpoint(weights_save, save_best_only=True, monitor='loss', mode='min')
        early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
        tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
        training_model.compile(optimizer=opt, loss='mae')
        training_model.fit(x=[x_train_rgb, x_train_feature], y=[y_train_rgb, y_train_feature],
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=1,
                        callbacks=[mcp_save, reduce_lr, early_stop, tensorboard_cb],
                        validation_data=([x_val_rgb, x_val_feature], [y_val_rgb, y_val_feature]))

        # Cleanup
        training_model.reset_states()
        K.clear_session()
        del train_data, x_train_rgb, x_train_feature, y_train_rgb, y_train_feature
        del val_data, x_val_rgb, x_val_feature, y_val_rgb, y_val_feature, training_model


    def predict(
            self,
            filepath: str,
            diff_weights: str='weights/diffuse.h5',
            spec_weights: str='weights/specular.h5',   
            overlap: int=32,         
            ): 
        
        """ Denoises supplied exr or all valid exr files in the given directory.

            filepath: Path to exr or directory of exrs to denoise
            overlap: Pixel overlap per denoising sample
            diff_weights: Path to diffuse weights .h5 file
            spec_weights: Path to specular weights .h5 file """

        # Instantiate diffuse and specular models and initialize their weights
        diff_denoiser, spec_denoiser  = self.build_model(), self.build_model()
        diff_denoiser.load_weights(diff_weights)
        spec_denoiser.load_weights(spec_weights)

        # Preprocess the exr(s)
        data = build_data(filepath, predict=True)

        diffuse, specular, name = data['noisy_diff'], data['noisy_spec'], data['name']
        albedo, emission, alpha = data['albedo'], data['emission'], data['alpha']

        # Denoise each exr and save the predictions to disk
        for i in range(len(diffuse)):
            n = name[i].split('/')[-1]
            pad = ' ' * 35
            print(f'\rDenoising {n}...' + pad)
            denoised_diffuse, denoised_specular = denoise_image(diffuse[i], specular[i], diff_denoiser, spec_denoiser, overlap=overlap)
            output_denoised(denoised_diffuse, denoised_specular, albedo[i], emission[i], alpha[i], name[i])

