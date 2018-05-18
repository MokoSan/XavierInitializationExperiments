import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.callbacks import History
from data.create_data import CreateDataset
from models.model import FullyConnectedNN
from argparse import ArgumentParser
import deepdish as dd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

max_gpus = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])

def build_parser():
    
    parser = ArgumentParser()
    
    parser.add_argument('--activation', type=str,
                        dest='activation', help='activation function for model',
                        default='tanh')

    parser.add_argument('--five_layer', action='store_true',
                        dest='five_layer', help='4 or 5 layer deep model')

    parser.add_argument('--num_gpus', type=int,
                        dest='num_gpus', help='number of gpus',
                        metavar='num_gpus', default=0)

    parser.add_argument('--xavier', dest='normalization', action = 'store_true',
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
    assert opts.num_gpus >= 0 and opts.num_gpus <= max_gpus
    assert opts.five_layer in [True, False]
    assert opts.normalization in [True, False]
    assert opts.dataset in ['mnist', 'cifar10', 'shapeset']
    assert opts.lr > 0
    assert opts.batch_size > 0
    assert opts.debug in [True, False]
    
def multi_gpu(model, num_gpus):
    
    with tf.device('/cpu:0'):
        p_model = model
    
    parallel_model = multi_gpu_model(p_model, gpus=num_gpus)
    
    return parallel_model
    
    
def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)
    
    data = CreateDataset()
    
    create_data_method = getattr(data, options.dataset)
    
    train_data, validation_data, params = create_data_method()
        
    kwargs = {"input_shape" : params['input_shape'], 
              "classes" : params['num_classes'], 
              "five_layers" : options.five_layer, 
              "activation" : options.activation,
              "normalization" : options.normalization}
    
    model = FullyConnectedNN(**kwargs)
    
    if options.num_gpus > 1:
        model = multi_gpu(model=model, num_gpus=options.num_gpus)
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=SGD(lr=options.lr))
    
    model.summary()
    
    num_epochs = params['num_epochs']
           
    history = History()
        
    if options.dataset == 'shapeset':    
        steps_per_epoch=params['steps_per_epoch']
        
        if options.debug:
            num_epochs = 5
            steps_per_epoch = 1000
        
        model.fit_generator(generator=train_data, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                            validation_data=validation_data, callbacks=[history])
    else:
        x, y = train_data
        
        if options.debug:
            num_epochs = 5
            x = x[:1000]
            y = y[:1000]
        
        model.fit(x=x, y=y, batch_size=options.batch_size, epochs=num_epochs, 
                  callbacks=[history], validation_data=validation_data)
    
    print 'COMPLETE'
        
if __name__ == '__main__':
    main()