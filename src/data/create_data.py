from tensorflow.python.keras import datasets
from tensorflow.python.keras.utils import to_categorical
import numpy as np
import itertools
import types
import sys
sys.path.append('../')
from lib.Shapeset.curridata import *
from lib.Shapeset.buildfeaturespolygon import *
from lib.Shapeset.polygongen import *


class KerasDataSetCreator(object):

    """
    CreateDataset class contains load_data method to create dataset for experiments. It takes advantages of prebuilt 
    Keras functions to load_data. 
    
    Approriate flags are mnist or cifar10

    The methods of this class return the training and validation sets for each dataset. The training set and validation set
    features are mean subtracted and scaled with standard deviation of the training dataset. 

    debug is used to create smaller validation set to ensure code works correctly
    """
    
    def __init__(self, flag='mnist', debug=False):
        self.flag = flag
        self.debug = debug
        
    def load_data(self):
        
        """
        Returns:
            - train_data: a tuple of numpy arrays for validation data features and one hot encoded classes         
            - validation_data: a tuple of numpy arrays for validation data features and one hot encoded classes
            - params: dictionary of parameters for running experiment i.e. input_shape, num_classes, num_epochs
        """
        
        dataset_class = getattr(datasets, self.flag)
        
        (x_train, y_train), (x_val, y_val) = dataset_class.load_data()
        
        x_mean = x_train.mean()
        x_std = x_train.std()

        x_train = x_train - x_mean
        x_train /= x_std
        x_train = x_train.reshape(x_train.shape[0], -1)
        y_train = to_categorical(y_train, 10)

        x_val = x_val - x_mean
        x_val /= x_std
        x_val = x_val.reshape(x_val.shape[0], -1)
        y_val = to_categorical(y_val, 10)
        
        num_examples_train = x_train.shape[0]
        num_examples_val = x_val.shape[0]
        
        num_feed_train = range(num_examples_train)
        num_feed_val = range(num_examples_val)
                
        params = {'input_shape': x_train.shape[1], 
                  'num_classes': y_train.shape[1],
                  'num_epochs': int(2.5e7/num_examples_train)}

        if self.debug: 
            params['num_epochs'] = 5
            num_feed_train = range(1000)
            num_feed_val = range(100)
            
        train_data = (x_train[num_feed_train], y_train[num_feed_train])
        validation_data = (x_val[num_feed_val], y_val[num_feed_val])
        
        return train_data, validation_data, params
        


class ShapesetDataCreator(object):
    
    """
    ShapesetDataCreator class contains method load_data to create dataset for experiment. This method uses the 
    Shapeset API created by glorotxa

    The method of this class returns the training and validation datasets and parameters for the experiment 
    The training data is a generator that yields features and classes.

    debug is used to create smaller validation set to ensure code works correctly
    """
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def load_data(self):

        """
        Returns:
            A generator that yeilds batches of size 10, of features and one hot encoded classes, and a validation set of
            300 examples
            - train_data: Infinite generator that yeilds batches of features and classes, batch size = 10
            - validation_data: a tuple of numpy arrays for validation data features and classes
            - params: dictionary of parameters for running experiment 
                      i.e. input_shape, num_classes, num_epochs, steps_per_epoch
        """

        genparams = {'inv_chance': 0.5, 'img_shape': (32, 32), 'n_vert_list': [3, 4, 20], 'fg_min': 0.55, 'fg_max': 1.0,
                     'bg_min': 0.0, 'bg_max': 0.45, 'rot_min': 0.0, 'rot_max': 1, 'pos_min': 0, 'pos_max': 1,
                     'scale_min': 0.2, 'scale_max': 0.8, 'rotation_resolution': 255, 'nb_poly_max': 2, 'nb_poly_min': 1,
                     'overlap_max': 0.5, 'poly_type': 2, 'rejectionmax': 50, 'overlap_bool': True}

        datagenerator = Polygongen
        funclist = [buildimage, buildedgesangle, builddepthmap, buildidentity, buildsegmentation, output,
                    buildedgesanglec]
        dependencies = [None, {'segmentation': 4}, None, None, {'depthmap': 2}, None, {'segmentation': 4}]
        funcparams = {'neighbor': 'V8', 'gaussfiltbool': False, 'sigma': 0.5, 'size': 5}
        nfeatures = 6

        batchsize = 10
        seed = 0
        funcparams.update({'neg': True})

        curridata = Curridata(nfeatures, datagenerator, genparams, funclist, dependencies, funcparams, batchsize, seed)

        train_data = self._generator(data_generator=curridata, batch_size=batchsize)
        x_val, y_val = self._create_validation_set(data_generator=curridata)
        
        num_feed_val = range(x_val.shape[0])
        
        params = {'input_shape': x_val.shape[1], 
                  'num_classes': y_val.shape[1],
                  'num_epochs': int(2.5e7/2e5),
                  'steps_per_epoch': 20000}
        
        if self.debug:
            params['num_epochs'] = 5
            params['steps_per_epoch'] = 1000
            num_feed_val = range(100)
            
        validation_data = (x_val[num_feed_val], y_val[num_feed_val])

        return train_data, validation_data, params

    def _generator(self, data_generator, batch_size):

        """
        Creates generator of batches of Shapeset features and labels, of batch size 10

        Input:
            - data_generator: curridata generator using the Shapeset API
            - batch_size: number of batches to be generated

        Returns:
            - batch_feautures: flattened batches of Shapeset features as numpy arrays
            - batch_labels: batches of one hot encoded Shapeset classes as numpy arrays
        """

        while True:
            data_generator.next()
            data_generator.image

            batch_features = data_generator._Curridata__features[0]
            batch_labels = self._convertout(data_generator.output)
            batch_labels = to_categorical(batch_labels, 9)

            yield batch_features, batch_labels

    def _convertout(self, out):

        target = 0 * ((out[:, 0] == 1) * (out[:, 1] == 0) * (out[:, 2] == 0)) + \
                 1 * ((out[:, 0] == 0) * (out[:, 1] == 1) * (out[:, 2] == 0)) + \
                 2 * ((out[:, 0] == 0) * (out[:, 1] == 0) * (out[:, 2] == 1)) + \
                 3 * ((out[:, 0] == 1) * (out[:, 1] == 1) * (out[:, 2] == 0)) + \
                 4 * ((out[:, 0] == 0) * (out[:, 1] == 1) * (out[:, 2] == 1)) + \
                 5 * ((out[:, 0] == 1) * (out[:, 1] == 0) * (out[:, 2] == 1)) + \
                 6 * ((out[:, 0] == 2) * (out[:, 1] == 0) * (out[:, 2] == 0)) + \
                 7 * ((out[:, 0] == 0) * (out[:, 1] == 2) * (out[:, 2] == 0)) + \
                 8 * ((out[:, 0] == 0) * (out[:, 1] == 0) * (out[:, 2] == 2))

        return target

    def _create_validation_set(self, data_generator):

        """
        Creates validaiton set by generator 30 batches of examples

        Input:
            - data_generator: curridata generator using the Shapeset API

        Returns:
             - x_val: validation set of flattened Shapeset features as numpy arrays
             - y_val: validation set of one hot encoded Shapeset classes as numpy arrays
        """

        x_val, y_val = [], []

        for _ in range(30):
            data_generator.next()
            data_generator.image

            x_val.append(data_generator._Curridata__features[0])
            y_val.append(to_categorical(self._convertout(data_generator.output), 9))

        x_val = list(itertools.chain(*x_val))
        y_val = list(itertools.chain(*y_val))

        return np.array(x_val), np.array(y_val)
    


if __name__ == '__main__':
    
    flag = 'cifar10'
    debug = True
    
    if flag == 'shapeset':
        data_class = ShapesetDataCreator(debug=debug)
    else:
        data_class = KerasDataSetCreator(flag=flag, debug=debug)
    
    train_data, validation_data, params = data_class.load_data()

    x_val, y_val = validation_data
        
    if isinstance(train_data, types.GeneratorType):
        x_train, y_train = next(train_data)  
    else:   
        x_train, y_train = train_data
    
    print (x_train.shape, y_train.shape, x_val.shape, y_val.shape, params)