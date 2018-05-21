from argparse import ArgumentParser
import numpy as np
#import matplotlib.mlab as mlab
#import matplotlib.pyplot as plt
import deepdish as dd
import scipy.stats as stats
#from matplotlib.pyplot import cm 
import os

history_DIR = 'model_history'


def build_parser():
    
    parser = ArgumentParser()
    
    parser.add_argument('--activation_type', type=str,
                        dest='activation', help='activation function for model',
                        default='tanh')

    parser.add_argument('--five_layer', action='store_true',
                        dest='five_layer', help='4 or 5 layer deep model')
    
    parser.add_argument('--gradients', action='store_true',
                        dest='gradients', help='Visualize Gradients or Activations', default=False)
    
    return parser

def check_opts(opts):
    assert opts.activation in ['sigmoid', 'tanh', 'softsign']
    assert opts.five_layer in [True, False]

def filename(opts):
    
    act = opts.activation
    
    if opts.five_layer:
        layers = 'five_layers'
    else:
        layers = 'four_layers'
        
    if opts.gradients:
        file = 'gradients.h5'
    else:
        file = 'activations.h5'
        
    f = []
    for norm in ['random', 'xavier']:
        fname = '-'.join([act,norm,layers])
        experiments = os.path.join(history_DIR, 'shapeset', fname, file)
                         
    return f

def plot_histogram(filename, index, ax=None):
    
    if ax is None:
        ax = plt.gca()
    
    experiment = dd.io.load(filename)

    color = ['r', 'g', 'b', 'c', 'k']
    
    x_min = []
    x_max = []
    y_max = []
    
    for i in range(len(experiment[index])):

        k = index

        noise = np.array(experiment[k][i]).reshape(-1, 1)
        mu = np.mean(noise)
        noise_norm = noise - mu
        mu_norm = np.mean(noise_norm)
        sigma_norm = np.std(noise_norm)

        X = np.linspace(mu_norm-3*sigma_norm, mu_norm+ 3*sigma_norm, 1000)
        Y = stats.norm.pdf(X, mu_norm, sigma_norm)
        ax.plot(X, Y, color=color[i], linestyle='-', label = 'Layer '+str(i+1))
        x_min.append(np.min(X))
        x_max.append(np.max(X))
        y_max.append(np.max(Y))
        
    ax.set_ylim(ymin=0, ymax=int(np.max(y_max))+1)
        
    if 'activation' in filename:
        ax.set_xlabel('Activation value')
        ax.set_xlim([-1, 1])
    else:
        ax.set_xlabel('Backpropogated gradients')
        ax.set_xlim(xmax = np.max(x_max), xmin=np.min(x_min))
    
    return ax

def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)
    act, norm, layers = filename(options)
    experiment_DIR = os.path.join(history_DIR, 'shapeset', fname)
    
        
    try:
        os.path.exists(experiment_DIR)
    except ValueError:
        print ('Experiment for Shapeset dataset using model with ' + ' '.join(layers.split('_')) + 
               ', ' + act + ' activation and ' + norm + ' normalization has not been run yet.')
    
if __name__ == '__main__':
    main()