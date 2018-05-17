from tensorflow.python.client import device_lib
from data.create_data import CreateDataset
from model import FullyConnectedNN
from argparse import ArgumentParser


max_gpus = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--activation', type=str,
                        dest='activation', help='activation function for model',
                        metavar='activation', default='sigmoid')

    parser.add_argument('--5layer', type=bool,
                        dest='five_layer', help='4 or 5 layer deep model',
                        metavar='five_layer', default=True)

    parser.add_argument('--num_gpus', type=int,
                        dest='num_gpus', help='number of gpus',
                        metavar='num_gpus', default=1)

    parser.add_argument('--xavier', type=bool,
                        dest='normalization', 
                        help='xavier or random initialization',
                        metavar='normalization', default=True)

    parser.add_argument('--dataset', type=str,
                        dest='dataset', 
                        help='dataset to use, mnist, cifar10, shapeset',
                        metavar='dataset', default='mnist')
    return parser

def check_opts(opts):
    assert opts.activation in ['sigmoid', 'tanh', 'softsign']
    assert opts.num_gpus > 0 and opts.num_gpus <= max_gpus
    assert opts.five_layer in [True, False]
    assert opts.normalization in [True, False]
    assert opts.dataset in ['mnist', 'cifar10', 'shapeset']

def main():
    pass
    
