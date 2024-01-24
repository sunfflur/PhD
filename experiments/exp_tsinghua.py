import jax
from jax import lax, numpy as jnp
from jax.example_libraries import optimizers
from typing import Any, Callable, Sequence, Optional
from flax import linen as nn
import optax
from flax.training import train_state
from flax.core import frozen_dict
import matplotlib.pyplot as plt
from main_functions.utils import *
from data_tsinghua import get_data

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


"""   
    Dataset from: http://bci.med.tsinghua.edu.cn/
    Starting with subject 1: S1.mat

"""
datapath = (
    "/home/natalia/Documentos/Dados/Tsinghua BCI Lab - Benchmark Dataset/"
)

"""
    Load data from from different stimulus frequency: 8, 10, 12 and 15 Hertz and
    choosing 16 electrodes (O1, O2, Oz, POz, Pz, PO3, PO4, PO7, PO8, P1, P2, Cz, C1, C2, CPz, FCz),
    following Vargas et. al (2022).

"""

sel_electrodes = {
    60: "O1",
    61: "Oz",
    62: "O2",
    55: "POz",
    47: "Pz",
    54: "PO3",
    56: "PO4",
    52: "PO7",
    58: "PO8",
    46: "P1",
    48: "P2",
    27: "Cz",
    26: "C1",
    28: "C2",
    37: "CPZ",
    18: "FCz",
}
stimulif = [8, 10, 12, 15]  # [10,15]#

subjects = [1]

x_train, x_val, x_test, y_train, y_val, y_test = get_data(
    datapath, sel_electrodes, stimulif, subjects
)


class FreqLayer(nn.Module):
    features: int
    # kernel_init: Callable = nn.initializers.normal()
    kernel_init: Callable = my_init

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel",
            self.kernel_init,  # Initialization function
            (x.shape[-1], 1),
        )  # shape info.
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
    bias_init: Callable = nn.initializers.normal()

    def setup(self):
        # Create the modules we need to build the network
        # nn.Dense is a linear layer
        self.linear1 = FreqLayer(features=1, name="freq1")
        self.linear2 = nn.Dense(
            features=4,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="dense2",
        )
        self.linear3 = nn.Dense(
            features=8,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="dense3",
        )
        self.linear4 = nn.Dense(features=4, name="dense4")

        # create the dropout modules
        self.droput1 = nn.Dropout(0.25, deterministic=True)
        self.droput2 = nn.Dropout(0.15, deterministic=True)

    # @nn.compact  # Tells Flax to look for defined submodules
    def __call__(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = nn.leaky_relu(x)
        x = self.droput1(x)
        x = self.linear3(x)
        x = nn.leaky_relu(x)
        x = self.droput2(x)
        x = self.linear4(x)
        return x


# B. Loss function we want to use for the optimization
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
def create_train_state(key, lr=0.001):  # adam: 1e-4
    """Creates initial `TrainState for our classifier.

    Args:
        key: PRNG key to initialize the model parameters
        lr: Learning rate for the optimizer

    """
    # 1. Model instance
    model = SimpleClassifier()

    # 2. Initialize the parameters of the model
    params = model.init(key, jnp.ones([1, x_train.shape[1]]))["params"]
    # print(jax.tree_map(lambda x: x.shape, params))

    # 3. Define the optimizer with the desired learning rate

    # 0.01, 0.09, x_train.shape[0]/BATCH_SIZE
    lrd = optimizers.inverse_time_decay(
        step_size=lr,
        decay_steps=x_train.shape[0] / BATCH_SIZE,
        decay_rate=0.05,
    )
    opt1 = optax.sgd(learning_rate=lrd, momentum=0.0)
    opt2 = optax.sgd(learning_rate=lr, momentum=0.9)
    opt3 = optax.adam(learning_rate=lrd)

    optimizer = optax.multi_transform(
        {"freq-opt": opt1, "d-opt": opt2},
        {
            "freq1": "freq-opt",
            "dense2": "d-opt",
            "dense3": "d-opt",
            "dense4": "d-opt",
        },
    )

    # 4. Create and return initial state from the above information. The `Module.apply` applies a
    # module method to variables and returns output and modified variables.
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=opt3
    )


EPOCHS = 200
BATCH_SIZE = 10

key = jax.random.PRNGKey(device)

key, init_key = jax.random.split(key)
state = create_train_state(init_key)

# Lists to record loss and accuracy for each epoch
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []

# Training loop
for i in range(EPOCHS):
    num_train_batches = len(x_train) // BATCH_SIZE
    num_valid_batches = len(x_val) // BATCH_SIZE
    # Lists to store loss and accuracy for each batch
    train_batch_loss, train_batch_acc = [], []
    valid_batch_loss, valid_batch_acc = [], []

    # Key to be passed to the data generator for augmenting
    # training dataset
    key, subkey = jax.random.split(key)
    train_indices = jax.random.permutation(
        subkey, jnp.arange(x_train.shape[0])
    )
    val_indices = jax.random.permutation(subkey, jnp.arange(x_val.shape[0]))
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

    print(
        f"loss: {epoch_train_loss:.3f}   acc: {epoch_train_acc:.3f}  valid_loss: {epoch_valid_loss:.3f}  valid_acc: {epoch_valid_acc:.3f}"
    )


# Let's plot the training and validataion losses as well as
# accuracies for both the dataset.
_, ax = plt.subplots(1, 2, figsize=(15, 8))
ax[0].plot(range(1, EPOCHS + 1), training_loss)
ax[0].plot(range(1, EPOCHS + 1), validation_loss)
ax[0].set_xlabel("Epochs")
ax[0].legend(["Training loss", "Validation loss"])

ax[1].plot(range(1, EPOCHS + 1), training_accuracy)
ax[1].plot(range(1, EPOCHS + 1), validation_accuracy)
ax[1].set_xlabel("Epochs")
ax[1].legend(["Training accuracy", "Validation accuracy"])

plt.show()
