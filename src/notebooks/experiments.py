
# coding: utf-8

# In[ ]:


from keras.objectives import categorical_crossentropy
from keras.callbacks import Callback, History
from keras.models import Sequential
from keras.layers import Dense, Input
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras import initializers, optimizers
from keras.utils.training_utils import multi_gpu_model
from keras.models import Model
from keras.utils import np_utils
import deepdish as dd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[ ]:


class GradientActivationStore(Callback):
    
    def __init__(self, filename):
        super(GradientActivationStore, self).__init__()
        self.gradients = []
        self.activations = []
        self.filename = filename
        self.y_true = K.placeholder(shape=[None, 9])
        
    def get_gradients(self, model):
        model_weights = model.trainable_weights
        weights = [weight for weight in model_weights if 'kernel' in weight.name]
        loss = K.mean(categorical_crossentropy(self.y_true, model.output))
        func = K.function([model.input, self.y_true], K.gradients(loss, weights))
        return func
    
    def get_activations(self, model):
        func = K.function([model.input], [layer.output for layer in model.layers[1:]]) # evaluation function
        return func

    def on_epoch_end(self, epoch, logs=None):
        #Gradients
        get_grad = self.get_gradients(self.model)
        inputs = [self.validation_data[0], self.validation_data[1]]
        grads = get_grad(inputs)
        self.gradients.append(grads)
        
        #Activations
        get_act = self.get_activations(self.model)
        acts = get_act([self.validation_data[0]])
        self.activations.append(acts)
        
    def on_train_end(self, logs=None):
        dd.io.save(self.filename+'-gradients.h5', self.gradients)
        dd.io.save(self.filename+'-activations.h5', self.activations)
        print ('COMPLETE')


# In[ ]:


def create_model(activation_func, depth, out_shape, in_shape=1024, N=True):
    
    w_in = np.sqrt(0.001)
    initializer = initializers.RandomUniform(minval=-w_in, maxval=w_in)
    
    if N:
        initializer = initializers.glorot_normal()
    
    x_in = Input(shape=(in_shape,))
    l = Dense(1000, activation=activation_func, kernel_initializer=initializer)(x_in)
    for i in range(2+depth):
        l = Dense(1000, activation=activation_func, kernel_initializer=initializer)(l)
    output = Dense(out_shape, activation='softmax', kernel_initializer=initializer)(l)

    return Model(inputs=x_in, outputs=output)


# In[ ]:


import sys
sys.path.insert(0, '/home/joshi/xavier/src/data/')
from Shapeset.curridata import *
from Shapeset.buildfeaturespolygon import *
from Shapeset.polygongen import *

genparams = {'inv_chance' : 0.5, 'img_shape' : (32,32), 'n_vert_list' : [3,4,20], 'fg_min' : 0.55, 'fg_max' : 1.0,        'bg_min' :0.0, 'bg_max': 0.45, 'rot_min' : 0.0, 'rot_max' : 1, 'pos_min' : 0, 'pos_max' : 1,         'scale_min' : 0.2, 'scale_max':0.8, 'rotation_resolution' : 255,        'nb_poly_max' :2, 'nb_poly_min' :1, 'overlap_max' : 0.5, 'poly_type' :2, 'rejectionmax' : 50,        'overlap_bool':True}

datagenerator=Polygongen
funclist =[buildimage,buildedgesangle,builddepthmap,buildidentity,buildsegmentation,output,buildedgesanglec]
dependencies = [None,{'segmentation':4},None,None,{'depthmap':2},None,{'segmentation':4}]
funcparams={'neighbor':'V8','gaussfiltbool' : False, 'sigma' : 0.5 , 'size': 5}
nfeatures = 6

batchsize = 10
seed = 0 
funcparams.update({'neg':True}) 
 
def convertout(out):
    target =    0*((out[:,0]==1) * (out[:,1] == 0) * (out[:,2]==0)) +                1*((out[:,0]==0) * (out[:,1] == 1) * (out[:,2]==0)) +                2*((out[:,0]==0) * (out[:,1] == 0) * (out[:,2]==1)) +                3*((out[:,0]==1) * (out[:,1] == 1) * (out[:,2]==0)) +                4*((out[:,0]==0) * (out[:,1] == 1) * (out[:,2]==1)) +                5*((out[:,0]==1) * (out[:,1] == 0) * (out[:,2]==1)) +                6*((out[:,0]==2) * (out[:,1] == 0) * (out[:,2]==0)) +                7*((out[:,0]==0) * (out[:,1] == 2) * (out[:,2]==0)) +                8*((out[:,0]==0) * (out[:,1] == 0) * (out[:,2]==2))
    return target

labels_map = ['Triangle', 'Square', 'Ellipse', 'Triangle and Square',
             'Square and Ellipse', 'Triangle and Ellipse', '2 Triangles',
             '2 Squares', '2 Ellipses']

curridata=Curridata(nfeatures,datagenerator,genparams,funclist,dependencies,funcparams,batchsize,seed)


# In[ ]:


def convertout(out):
    target =    0*((out[:,0]==1) * (out[:,1] == 0) * (out[:,2]==0)) +                1*((out[:,0]==0) * (out[:,1] == 1) * (out[:,2]==0)) +                2*((out[:,0]==0) * (out[:,1] == 0) * (out[:,2]==1)) +                3*((out[:,0]==1) * (out[:,1] == 1) * (out[:,2]==0)) +                4*((out[:,0]==0) * (out[:,1] == 1) * (out[:,2]==1)) +                5*((out[:,0]==1) * (out[:,1] == 0) * (out[:,2]==1)) +                6*((out[:,0]==2) * (out[:,1] == 0) * (out[:,2]==0)) +                7*((out[:,0]==0) * (out[:,1] == 2) * (out[:,2]==0)) +                8*((out[:,0]==0) * (out[:,1] == 0) * (out[:,2]==2))
    return target

def generator(data_generator, batch_size):
 # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, 1024))
    batch_labels = np.zeros((batch_size, 9))
    
    while True:
        
        data_generator.next()
        data_generator.image
        
        batch_features = data_generator._Curridata__features[0]
        batch_labels = convertout(data_generator.output)
        batch_labels = np_utils.to_categorical(batch_labels, 9)
        
        yield batch_features, batch_labels

L = generator(curridata, 10)

#VALIDATION DATA 

x_val, y_val = [], []

for i in range(30):
    curridata.next()
    curridata.image
    x_val.append(curridata._Curridata__features[0])
    y_val.append(np_utils.to_categorical(convertout(curridata.output), 9))
    
    import itertools

x_val = list(itertools.chain(*x_val))
y_val = list(itertools.chain(*y_val))

x_val = np.array(x_val)
y_val = np.array(y_val)

print x_val.shape, y_val.shape


# In[ ]:


def prepare_data(L, x_val, y_val):
    return L, x_val, y_val


# In[ ]:


def experiment(L, x_val, y_val, dataset, activation, Nn, depth, num_gpus=1):
    
    K.clear_session()
    
    #model = multi_gpu_model(create_model(activation_func=activation, depth=depth, out_shape=9, N=Nn), gpus=num_gpus)
    
    model = create_model(activation_func=activation, depth=depth, out_shape=9, N=Nn)
    
    Lk, x_valk, y_valk = prepare_data(L, x_val, y_val)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=0.001),
                  metrics=['accuracy'])
    
    if Nn:
        name = dataset + '-' + activation + '-N' + '-' + str(depth) + '-history'
    else:
        name = dataset + '-' + activation + '-notN' + '-' + str(depth) + '-history'
        
    filename = name + '.h5'
    
    history = History()
    
    cbk = GradientActivationStore('../logs/'+name)
    
    model.fit_generator(Lk, steps_per_epoch=20000, epochs=125, 
                    callbacks=[history, cbk], validation_data=(x_valk, y_valk))

    dd.io.save('../model_history/shapeset-pickles/'+filename, history.history)
    
    return filename


# In[ ]:


filename = experiment(L, x_val, y_val, 'shapeset', 'sigmoid', True, 0, num_gpus=1)
filename = experiment(L, x_val, y_val, 'shapeset', 'sigmoid', False, 0, num_gpus=1)
filename = experiment(L, x_val, y_val, 'shapeset', 'sigmoid', True, 1, num_gpus=1)
filename = experiment(L, x_val, y_val, 'shapeset', 'sigmoid', False, 1, num_gpus=1)
filename = experiment(L, x_val, y_val, 'shapeset', 'softsign', True, 1, num_gpus=1)
filename = experiment(L, x_val, y_val, 'shapeset', 'softsign', False, 1, num_gpus=1)
filename = experiment(L, x_val, y_val, 'shapeset', 'tanh', True, 1, num_gpus=1)
filename = experiment(L, x_val, y_val, 'shapeset', 'tanh', False, 1, num_gpus=1)

