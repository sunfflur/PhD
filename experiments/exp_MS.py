from pathlib import Path
from itertools import product
import os
import jax
from jax import lax, numpy as jnp
import numpy as np
import pandas as pd
from jax.example_libraries import optimizers
from typing import Any, Callable, Sequence, Optional
from flax import linen as nn
import optax
from flax.training import train_state
from flax.core import frozen_dict
import matplotlib.pyplot as plt
from main_functions.utils import *
#from main_functions.dataloader import MotionSense_dataloader
#from main_functions.preprocessing import mnist_preprocessing
from data_MS import get_data
from functools import partial
import time
np.random.seed(56)


# Specify GPU device
gpu_devices = jax.devices("gpu")
if not gpu_devices:
    raise RuntimeError("No GPU devices found.")
else:
    print("GPU found!")
device = gpu_devices[0].id
print(f"Using device: {jax.devices()[0].device_kind}")

get_gpu_memory_info()

# define the file path for the original data
datapath = '/home/natalia/Git_Projects/PhD/data/DAGHAR/MotionSense/'

KFold = True
EPOCHS = 500 #maybe implement early stopping
#BATCH_SIZE= 64

if KFold == True:
    print("Running MotionSense DAGHAR dataset")
    
    # Grid-search K-fold - running for 1 subject
    classes = [0,1,2,3,4,5]
    n_classes = len(classes)
    #subjects = [1,2,3,4,5,6] #1
    learning_rates = [0.001, 0.004, 0.01, 0.0001] # DHT=[0.01]
    opts = ["opt1","opt3", "opt4", "opt5", "opt7", "opt8"]
    neurons = [[2, 4], [4, 4], [4, 8], [8, 16], [16, 16], [16, 32], [32, 32]]
    levels_list = [1, 2] 
    band_widths = [1, 2, 3]
    functions = ['DHT', 'DFT'] # ['halfDHT', 'halfDFT']
    seconds_off = [0.0, 0.0] #[0, 0.5
    windows_overlaps = [[3, 0]] # windows and overlaps 
    dropout_taxes = [[0.2, 0.2], [0.5, 0.5], [0.3, 0.5], [0.6, 0.2], [0.30, 0.15]]
    freq_means = [2.0, 1.0, 0.0]
    freq_stds = [0.1, 0.01, 0.001]
    pooling = ['Sum', 'Mean', 'Max'] #'Sum'
    batch_sizes = [64, 128, 256]

    results = {
        'Function': [],
        'Epochs': [],
        'Batch_Size': [],
        'Pooling': [],
        'Dropout': [],
        'FreqMean': [],
        'FreqStd': [],
        'Time_Off': [],
        'Window': [],
        'Overlap': [],
        'Levels': [],
        'Band_Width': [],
        'Neuron_Configuration': [],
        'Optimizer': [],
        'Learning_Rate': [],
        'Val_Accuracy': [],
        'Time': [],
        'Params': [],
    }

    configs_list = []
    for BATCH_SIZE in batch_sizes:
        for pool in pooling:
            for fstd in freq_stds:
                for fmean in freq_means:
                    for dp in dropout_taxes:
                        for wo in windows_overlaps:
                            for off in seconds_off:
                                for f in functions:
                                    for levels in levels_list:
                                        for width in band_widths:
                                            for neuron_list in neurons:
                                                for opt in opts:
                                                    for lrs in learning_rates:
                                                        configs_list.append((levels, neuron_list, opt, lrs, f, off, wo[0], wo[1], width, dp[0], dp[1], fmean, fstd, pool, BATCH_SIZE))

    np.random.shuffle(configs_list)
    for config in configs_list:
        #Important
        neuron1 = 1
        neuron4 = n_classes
        levels = config[0]
        neuron2, neuron3 = config[1]
        opt = config[2]
        lrs = config[3]
        f = config[4]
        off = config[5]
        wo[0] = config[6]
        wo[1] = config[7]
        width = config[8]
        dropout_0 = config[9]
        dropout_1 = config[10]
        freq_mean = config[11]
        freq_std = config[12]
        pool = config[13]
        BATCH_SIZE = config[14]
        #End important

        # Identify the config to not repeat
        path_to_file = os.path.join(os.getcwd(), "experiments", "toys", f"MS_MultiSubject_{f}_{n_classes}_{pool}")
        Path.mkdir(Path(path_to_file), exist_ok=True, parents=True)
        filename = f"MultiSubject_{levels}_{width}_{neuron1}_{neuron2}_{dropout_0}_{neuron3}_{dropout_1}_{neuron4}_{opt}_{str(round(lrs,4))}_{off}_{wo[0]}_{wo[1]}_{BATCH_SIZE}"
        save_file_name = os.path.join(path_to_file,filename)
        if os.path.exists(save_file_name):
            print(f"{filename} already exists!")
            continue

        # Get the processed data
        key = jax.random.PRNGKey(device)
        key, init_key = jax.random.split(key)
        x_train, x_val, x_test, y_train, y_val, y_test = get_data(datapath, 
                                                                  sampling_frequency=20,
                                                                  n_levels=levels, 
                                                                  band_width=width,
                                                                  transform=f, 
                                                                  window=wo[0], 
                                                                  overlap=wo[1],
                                                                  pooling_type=pool)
        
        print("Batch Size: ", BATCH_SIZE)
        print("Starting... ")
        class FreqLayer(nn.Module):
            features: int
            # kernel_init: Callable = nn.initializers.normal()
            freq_mean: Callable = freq_mean
            freq_std: Callable = freq_std
            my_init2 = partial(my_init, mean=freq_mean, std=freq_std)
            kernel_init: Callable = my_init2

            @nn.compact
            def __call__(self, x):
                kernel = self.param(
                    "kernel",
                    self.kernel_init,  # Initialization function
                    (x.shape[-1], 1))  # shape info.
            
                y = x * jnp.ravel(kernel)  # (10,1000)
                # bias = self.param('bias', self.bias_init, (self.features,))
                # y = y + bias
                return y

        # Define the neural network model using FLAX
        class SimpleClassifier(nn.Module):
            """SimpleClassifier
            Define the neural network model using FLAX

            """

            # num_hidden: int
            # num_outputs: int
            # mean_value: float
            # features: int
            kernel_init: Callable = nn.initializers.glorot_normal()
            bias_init: Callable = my_bias_init
            neuron1: Callable = neuron1
            neuron2: Callable = neuron2
            neuron3: Callable = neuron3
            neuron4: Callable = neuron4
            dropout_0: Callable = dropout_0
            dropout_1: Callable = dropout_1

            def setup(self):
                # Create the modules we need to build the network
                # nn.Dense is a linear layer
                self.linear1 = FreqLayer(features=self.neuron1, name="freq1")
                self.linear2 = nn.Dense(
                    features=self.neuron2,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    name="dense2",
                )
                self.linear3 = nn.Dense(
                    features=self.neuron3,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    name="dense3",
                )
                self.linear4 = nn.Dense(features=self.neuron4, name="dense4")

                # create the dropout modules
                self.dropout1 = nn.Dropout(self.dropout_0, deterministic=True) #0.30
                self.dropout2 = nn.Dropout(self.dropout_1, deterministic=True) #0.15

            # @nn.compact  # Tells Flax to look for defined submodules
            def __call__(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = nn.leaky_relu(x)
                x = self.dropout1(x)
                x = self.linear3(x)
                x = nn.leaky_relu(x)
                x = self.dropout2(x)
                x = self.linear4(x)
                return x

        # B. Loss function we want to use for the optimization
        @jax.jit
        def calculate_loss(params, inputs, labels):
            """Cross-Entropy loss function.

            Args:
                params: The parameters of the model at the current step
                inputs: Batch of images
                labels: Batch of corresponding labels
            Returns:
                loss: Mean loss value for the current batch
                logits: Output of the last layer of the classifier
            """
            logits = SimpleClassifier().apply({"params": params}, inputs)
            loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
            return loss, logits

        # C. Evaluation metric
        @jax.jit
        def calculate_accuracy(logits, labels):
            """Computes accuracy for a given batch.

            Args:
                logits: Output of the last layer of the classifier
                labels: Batch of corresponding labels
            Returns:
                Mean accuracy for the current batch
            """
            return jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))

        # D. Train step. We will `jit` transform it to compile it. We will get a
        # good speedup on the subseuqent runs
        @jax.jit
        def train_step(state, batch_data):
            """Defines the single training step.

            Args:
                state: Current `TrainState` of the model
                batch_data: Tuple containing a batch of images and their corresponding labels
            Returns:
                loss: Mean loss for the current batch
                accuracy: Mean accuracy for the current batch
                state: Updated state of the model
            """

            # 1. Get the images and the labels
            inputs, labels = batch_data

            # 2. Calculate the loss and get the gradients
            (loss, logits), grads = jax.value_and_grad(calculate_loss, has_aux=True)(
                state.params, inputs, labels
            )

            # 3. Calculate the accuracy for the cuurent batch
            accuracy = calculate_accuracy(logits, labels)

            # 4. Update the state (parameters values) of the model
            state = state.apply_gradients(grads=grads)

            # 5. Return loss, accuracy and the updated state
            return loss, accuracy, state

        # E. Test/Evaluation step. We will `jit` transform it to compile it as well.
        @jax.jit
        def test_step(state, batch_data):
            """Defines the single test/evaluation step.

            Args:
                state: Current `TrainState` of the model
                batch_data: Tuple containingm a batch of images and their corresponding labels
            Returns:
                loss: Mean loss for the current batch
                accuracy: Mean accuracy for the current batch
            """
            # 1. Get the images and the labels
            inputs, labels = batch_data

            # 2. Calculate the loss
            loss, logits = calculate_loss(state.params, inputs, labels)
            # 3. Calculate the accuracy
            accuracy = calculate_accuracy(logits, labels)

            # 4. Return loss and accuracy values
            return loss, accuracy

        # F. Initial train state including parameters initialization
        @jax.jit
        def create_train_state(key, lr=lrs):  # adam: 0.001=1e-4
            global opt
            """Creates initial `TrainState for our classifier.

            Args:
                key: PRNG key to initialize the model parameters
                lr: Learning rate for the optimizer

            """
            # 1. Model instance
            model = SimpleClassifier()

            # 2. Initialize the parameters of the model
            params = model.init(key, jnp.ones([1, x_train.shape[1]]))["params"]

            # 3. Define the optimizer with the desired learning rate

            # 0.01, 0.09, x_train.shape[0]/BATCH_SIZE
            lrd = optimizers.inverse_time_decay(
                step_size=lr,
                decay_steps=x_train.shape[0] / BATCH_SIZE,
                decay_rate=0.05,
            )
            opt1 = optax.sgd(learning_rate=lrd, momentum=0.0)
            opt2 = optax.sgd(learning_rate=lrd, momentum=0.9)
            opt3 = optax.adam(learning_rate=lrd)
            opt4 = optax.adamw(learning_rate=float(lr))
            opt5 = optax.amsgrad(learning_rate=float(lr))
            opt6 = optax.multi_transform(
                {"freq-opt": opt1, "d-opt": opt2},
                {
                    "freq1": "freq-opt",
                    "dense2": "d-opt",
                    "dense3": "d-opt",
                    "dense4": "d-opt",
                },
            )
            opt7 = optax.adam(learning_rate=float(lr))
            opt8 = optax.sgd(learning_rate=float(lr), momentum=0.9)
            optim_dict = {"opt1":opt1,"opt2":opt2,"opt3":opt3,
                        "opt4":opt4,"opt5":opt5, "opt6":opt6, 
                        "opt7":opt7, "opt8":opt8}
            # 4. Create and return initial state from the above information. The `Module.apply` applies a
            # module method to variables and returns output and modified variables.
            #opt3 best option lr=0.001
            return train_state.TrainState.create(
                apply_fn=model.apply, params=params, tx=optim_dict[opt]
            )

        state = create_train_state(init_key)
        params_dict = jax.tree_util.tree_map(lambda x: x.shape, state.params)
        total_params = count_params(params_dict)

        # Lists to record loss and accuracy for each epoch
        training_loss = []
        validation_loss = []
        training_accuracy = []
        validation_accuracy = []

        # Training loop
        start = time.time()
        for i in range(EPOCHS):
            num_train_batches = len(x_train) // BATCH_SIZE
            num_valid_batches = len(x_val) // BATCH_SIZE
            # Lists to store loss and accuracy for each batch
            train_batch_loss, train_batch_acc = [], []
            valid_batch_loss, valid_batch_acc = [], []

            # Key to be passed to the data generator for augmenting
            # training dataset
            key, subkey = jax.random.split(key)
            train_indices = jax.random.permutation(subkey, jnp.arange(x_train.shape[0]))
            val_indices = jax.random.permutation(subkey, jnp.arange(x_val.shape[0]))
            if i%50==0:
                print(f"Epoch: {i+1:<3}", end=" ")

            # Training
            for step in range(num_train_batches):
                batch_indices = train_indices[
                    step * BATCH_SIZE : (step + 1) * BATCH_SIZE
                ]
                batch_data = (x_train[batch_indices], y_train[batch_indices])
                # batch_data = next(train_data_gen)
                loss_value, acc, state = train_step(state, batch_data)
                train_batch_loss.append(loss_value)
                train_batch_acc.append(acc)

            # Evaluation on validation data
            for step in range(num_valid_batches):
                batch_indices = val_indices[
                    step * BATCH_SIZE : (step + 1) * BATCH_SIZE
                ]
                batch_data = (x_val[batch_indices], y_val[batch_indices])
                # batch_data = next(valid_data_gen)
                loss_value, acc = test_step(state, batch_data)
                valid_batch_loss.append(loss_value)
                valid_batch_acc.append(acc)

            # Loss for the current epoch
            epoch_train_loss = jnp.mean(jnp.asarray(train_batch_loss))
            epoch_valid_loss = jnp.mean(jnp.asarray(valid_batch_loss))

            # Accuracy for the current epoch
            epoch_train_acc = jnp.mean(jnp.asarray(train_batch_acc))
            epoch_valid_acc = jnp.mean(jnp.asarray(valid_batch_acc))

            training_loss.append(epoch_train_loss)
            training_accuracy.append(epoch_train_acc)
            validation_loss.append(epoch_valid_loss)
            validation_accuracy.append(epoch_valid_acc)

            if i%50==0:
                print(
                    f"loss: {epoch_train_loss:.3f}   acc: {epoch_train_acc:.3f}  valid_loss: {epoch_valid_loss:.3f}  valid_acc: {epoch_valid_acc:.3f}"
                )
        end = time.time()
        training_time = end - start
        #print("--- %.2f seconds ---" % (training_time))
        
        def evaluation(x_val, y_val):
            # Select some samples randomly from the validation data
            #jnp.random.seed(0)
            random_idx = jax.random.choice(key, jnp.arange(len(x_val)), shape=(x_val.shape[0],), replace=False)
            random_valid_samples = x_val[random_idx], y_val[random_idx]

            # Get the predictions
            logits_test = SimpleClassifier().apply({'params': state.params}, random_valid_samples[0])

            # Calculate the accuracy for these samples
            acc_test = calculate_accuracy(logits_test, random_valid_samples[1])
            #predicted_class = jnp.asarray(jnp.argmax(logits_test, -1))

            print(f"Accuracy on randomly selected sample of size {len(random_idx)} is {acc_test*100:.2f} %\n")
            return acc_test
        #test_accuracie = evaluation(x_test, y_test) # not running yet
        val_accuracy = jnp.asarray(max(validation_accuracy))
        print("VAL ACCURACY:", val_accuracy)
        total_time = jnp.asarray(training_time)
        #mean_trials = jnp.mean(jnp.asarray(accuracies_per_trial))
        #times.append(total_time)
        #accuracies.append(mean_trials)

        data = {
            'Function': f,
            'Epochs': str(EPOCHS),
            'Batch_Size': str(BATCH_SIZE),
            'Pooling': pool,
            'Dropout': str((dropout_0,dropout_1)),
            'FreqMean': str(freq_mean),
            'FreqStd': str(freq_std),
            'Time_Off': str(off),
            'Window': str(wo[0]),
            'Overlap': str(wo[1]),
            'Levels': str(levels),
            'Band_Width': str(width),
            'Neuron_Configuration': str((neuron1,neuron2,neuron3,neuron4)),
            'Optimizer': opt,
            'Learning_Rate': str(lrs),
            'Val_Accuracy': str(f"{val_accuracy:.4f}"),
            'Time': str(f"{total_time:.2f}"),
            'Params': str(total_params)
        }
        df_cfg = pd.DataFrame.from_dict(data, orient="index").transpose()
        df_cfg.to_csv(save_file_name)
