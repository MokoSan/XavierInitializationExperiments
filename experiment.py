import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.callbacks import History
from lib.KerasHelpers.modelhelpers import GradientActivationStore, model_placement
from src.data.create_data import CreateDataset
from src.models.model import fully_connected_neural_net
from argparse import ArgumentParser
import deepdish as dd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def build_parser():
    
    parser = ArgumentParser()
    
    parser.add_argument('--activation', type=str,
                        dest='activation', help='activation function for model',
                        default='tanh')

    parser.add_argument('--is_five_layers', action='store_true',
                        dest='is_five_layers', help='4 or 5 layer deep model')

    parser.add_argument('--num_gpus', type=int,
                        dest='num_gpus', help='number of gpus. 0 for cpu only.',
                        metavar='num_gpus', default=0)

    parser.add_argument('--xavier', dest='is_normalized', action = 'store_true',
                        help='xavier or random initialization',default=False)

    parser.add_argument('--dataset', type=str, dest='dataset', 
                        help='dataset to use, mnist, cifar10, shapeset',
                        default='mnist')
    
    parser.add_argument('--lr', type=float, dest='lr', 
                        help='learning rate for training',
                        default=0.001)
    
    parser.add_argument('--batch_size', type=int, dest='batch_size', 
                        help='batch_Size for training',
                        default=32)
    
    parser.add_argument('--debug', dest='debug', action = 'store_true',
                        help='debug mode', default=False)
    
    return parser

def check_opts(opts):
    assert opts.activation in ['sigmoid', 'tanh', 'softsign']
    assert opts.num_gpus >= 0
    assert opts.is_five_layers in [True, False]
    assert opts.is_normalized in [True, False]
    assert opts.dataset in ['mnist', 'cifar10', 'shapeset']
    assert opts.lr > 0
    assert opts.batch_size > 0
    assert opts.debug in [True, False]

def filename(opts):

    act = opts.activation

    if opts.is_normalized:
        norm = 'xavier'
    else:
        norm = 'random'
    
    if opts.is_five_layers:
        layers = 'five_layers'
    else:
        layers = 'four_layers' 
    
    return '-'.join([act, norm, layers])
    
    
def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)
    fname = filename(options)
    
    data = CreateDataset()
    
    create_data_method = getattr(data, options.dataset)
    
    train_data, validation_data, params = create_data_method()
    
    history_DIR = 'model_history'
    data_DIR = os.path.join(history_DIR, options.dataset)
    experiment_DIR = os.path.join(data_DIR, fname)
    
    if (not os.path.exists(history_DIR) and not options.debug):
        os.makedirs(history_DIR)
    
    if (not os.path.exists(data_DIR) and not options.debug):
        os.makedirs(data_DIR)
        os.makedirs(experiment_DIR)
        
    kwargs = {"input_shape" : params['input_shape'], 
              "classes" : params['num_classes'], 
              "is_five_layers" : options.is_five_layers, 
              "activation" : options.activation,
              "is_normalized" : options.is_normalized}
    
    model = fully_connected_neural_net(**kwargs)
    
    p_model = model_placement(model=model, num_gpus=options.num_gpus)
    
    p_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=SGD(lr=options.lr))
    
    p_model.summary()
    
    num_epochs = params['num_epochs']
                       
    if options.dataset == 'shapeset':    
        steps_per_epoch=params['steps_per_epoch']
        
        history = History()
        GAStore = GradientActivationStore(DIR=experiment_DIR, 
                        num_classes=params['num_classes'], 
                        record_every=1, only_weights=True)
        
        callbacks = [history, GAStore]
        
        if options.debug:
            num_epochs = 5
            steps_per_epoch = 1000
            callbacks = None
        
        p_model.fit_generator(generator=train_data, 
                            steps_per_epoch=steps_per_epoch, 
                            epochs=num_epochs, 
                            validation_data=validation_data, 
                            callbacks=callbacks)
    else:
        x, y = train_data
        history = History()
        callbacks = [history]
        
        if options.debug:
            num_epochs = 5
            x = x[:1000]
            y = y[:1000]
            callbacks = None
        
        p_model.fit(x=x, y=y, 
                  batch_size=options.batch_size, 
                  epochs=num_epochs, 
                  callbacks=callbacks, 
                  validation_data=validation_data)
    
    if not options.debug:
        dd.io.save(os.path.join(experiment_DIR, 'history.h5'), history.history)    
            
if __name__ == '__main__':
    main()
