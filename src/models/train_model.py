import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.keras.utils import multi_gpu_model
from data.create_data import CreateDataset
from models.model import FullyConnectedNN
from argparse import ArgumentParser

max_gpus = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--activation', type=str,
                        dest='activation', help='activation function for model',
                        default='sigmoid')

    parser.add_argument('--five_layer', type=bool,
                        dest='five_layer', help='4 or 5 layer deep model',
                        default=True)

    parser.add_argument('--num_gpus', type=int,
                        dest='num_gpus', help='number of gpus',
                        metavar='num_gpus', default=0)

    parser.add_argument('--xavier', dest='normalization', action = 'store_true',
                        help='xavier or random initialization',default=False)

    parser.add_argument('--dataset', type=str, dest='dataset', 
                        help='dataset to use, mnist, cifar10, shapeset',
                        default='mnist')
    
    parser.add_argument('--debug', dest='debug', action = 'store_true',
                        help='debug mode', default=False)
    
    return parser

def check_opts(opts):
    assert opts.activation in ['sigmoid', 'tanh', 'softsign']
    assert opts.num_gpus >= 0 and opts.num_gpus <= max_gpus
    assert opts.five_layer in [True, False]
    assert opts.normalization in [True, False]
    assert opts.dataset in ['mnist', 'cifar10', 'shapeset']
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

    model.summary()
    
    
    
if __name__ == '__main__':
    main()