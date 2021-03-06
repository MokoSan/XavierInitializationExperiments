# xavier_init

This repo uses TensorFlow to replicate the experiments and results of ["Understanding the difficulty of training deep feedforward neural networks" by Xavier Glorot, Yoshua Bengio](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

## Implementation Details

Our implementations uses TensorFlow to train a feed forward neural network. The experiments compare Xaiver and Random Initialization for activation functions with fairly linear sections such as tanh, softsign and sigmoid. 

There are two experiments conducted with this code
* `Validation Error/Convergence`:  Models are trained on CIFAR10, MNIST and Shapeset with combinations of normalization schemes and activation functions. Error Rate and Loss on the validation set are recorded and compared. 

* `Gradients and Activation Study`: For the Shapeset dataset, gradients and activations are recorded on the validation set for combinations of normalization schemes and activation functions. The distributions of the parameters are studied to see the effect of activation and normalization on model weights. 

Our implementation did not tune for hyperparameters because of the large number of experiments that would be needed to run i.e 3 activation functions * 3 datasets * 2 normalization schemes * 2 architectures. Given the long training time, running 36 experiments can be very time consuming. Although, `experiment.py` contains flags for batch size and learning rate, if others would like to replicate our results. 

## Documentation

### Setup
Run `./install.sh` in the working directory. It installs all the requirements from `requirements.txt` into a virtual environment, and clones the `Shapeset` and `KerasHelpers git` repo in to lib/ folder. It also creates a jupyter kernel called `xavier` that allows jupyter notebooks to be created using the virtual environment. 

#### requirements.txt
* tensorflow-gpu
* deepdish
* scipy
* numpy
* pandas
* pygame
* matplotlib
* h5py==2.8.0rc1 (Until new h5py update)
* ipykernel

#### Shapeset
[Shapeset]("https://github.com/glorotxa/Shapeset") is repository created by Glorot Xavier. 

#### KerasHelpers
[KerasHelpers]("https://github.com/anmolsjoshi/KerasHelpers") is a repository containing helper functions for Keras. We used the GradientActivationStore Callback. 

### Running Experiments
Use experiment.py to train model using different activation functions, initialization schemes and datasets. Run `python experiment.py` to conduct experiment. Training takes about an hour on a GTX 1080 Ti. **Before you run this, you should run `./install.sh`**.

#### experiment.py
`experiment.py` runs experiment of image classification using feed forward networks on different initialization schemes, activation functions and datasets. It saves training history, which includes training and validation accuracy and loss, and gradients and activations at the end of each epoch using the validation set. At the end of training, three files are saved into `model_history/name_of_dataset/name_of_experiment/`:

* `history.h5` : h5py file containing a dictionary of the training and validation accuracy and loss. 
* `activations.h5`: h5py file containing a dictionary of the activations on the validation set at the beginning of training and at the end of each epoch. The keys are 'epoch%d' and that results in another dictionary with keys of layers names such as 'dense_%d'.
* `gradients.h5`: h5py file similar to activations.h5 but from backpropogated gradients.

##### Example:
    python experiment.py --xavier --activation tanh --five_layer --dataset cifar10 --num_gpus 1
          
##### Flags
* `--debug`: Debug mode. Runs model on reduced dataset for 5 epochs. Mode is used to ensure requirements and libraries are working correctly. Does not save history. 
* `--xavier`: Flag to use xavier initialization. If not called, random initialization is used.
* `--activation`: Activation function for model. sigmoid, tanh, softsign can be used. Default is tanh.
* `--five_layer`: Flag to use a five layer model. If not called, four layer model is used. 
* `--dataset`: Dataset to train network on. mnist, cifar10, shapeset can be used. Default is mnist. 
* `--lr`: Learning rate for model. Default 1e-3. 
* `--num_gpus`: Number of GPUs. Greater than 1 will run in parallel mode. 1 will use GPU. 0 will use CPU. Default is 0. 

