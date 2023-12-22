import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from main import x_train, x_val, x_test, y_train, y_val, y_test # data
import subprocess
from flax.training import train_state
#XLA_PYTHON_CLIENT_PREALLOCATE=False

# Specify GPU device
gpu_devices = jax.devices("gpu")
if not gpu_devices:
    raise RuntimeError("No GPU devices found.")
else:
    print("GPU found!")
device = gpu_devices[0].id
print(f"Using device: {jax.devices()[0].device_kind}")

def get_gpu_memory_info():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.used', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip().split('\n')

    for line in output:
        free_memory, used_memory = map(int, line.split(','))
        
        print(f"Free GPU Memory: {free_memory} MiB")
        print(f"Used GPU Memory: {used_memory} MiB")

get_gpu_memory_info()




# Define the frequency layer.

class FreqLayer(nn.Module):
    """Custom frequency layer."""
    mean_value: int
    
    @nn.compact
    def __call__(self, x):
        """Applies pointwise product to the input x."""
        # Assuming x has shape (batch_size, input_size)
        # Initialize weights with shape (input_size,)
        #print(x.shape)
        w = self.param('weights', nn.initializers.normal(stddev=0.1), (x.shape[1],))
        #print('w:', w)
        result = (x+self.mean_value) * w
        return result

# Define the neural network model using FLAX
class SimpleClassifier(nn.Module):
    """SimpleClassifier
    Define the neural network model using FLAX
    
    """
    num_hidden: int
    num_outputs: int 
    mean_value: float


    @nn.compact  # Tells Flax to look for defined submodules
    def __call__(self, x):
        # Perform the calculation of the model to determine the prediction
        # while defining necessary layers
        
        x = FreqLayer(mean_value=self.mean_value, name='freqlayer')(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(features=self.num_hidden, kernel_init=nn.initializers.glorot_normal(), bias_init=nn.initializers.normal())(x)
        #print('x shape:', x.shape)
        x = nn.leaky_relu(x)
        x = nn.Dropout(0.25, deterministic=True)(x)
        x = nn.Dense(features=self.num_hidden*2, kernel_init=nn.initializers.glorot_normal(), bias_init=nn.initializers.normal())(x)
        x = nn.leaky_relu(x)
        x = nn.Dropout(0.15, deterministic=True)(x)
        x = nn.Dense(features=self.num_outputs)(x)
        #x = nn.log_softmax(x)
        return x

# Define the loss and acc functions

def loss(params, batch):
    inputs, labels = batch
    logits = SimpleClassifier.apply(params, inputs) #model instead
    #return -jnp.mean(jnp.sum(logits * labels, axis=-1))
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
    return loss, logits
    #return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))


def accuracy(params, batch):
    inputs, targets = batch
    #print('targets', targets)
    logits = SimpleClassifier.apply(params, inputs) #model instead
    #predicted_labels = jnp.argmax(logits, axis=-1)
    #correct_predictions = jnp.sum(predicted_labels == jnp.argmax(targets))
    #total_samples = inputs.shape[0]
    #accuracy = correct_predictions / total_samples
    #return accuracy.item()
    return jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(targets, -1))

# Define a function for updating parameters using the optimizer
@jax.jit
def update(params, opt_state, batch):
    grads = jax.grad(loss)(params, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


@jax.jit
def train_step(state, batch_data):
    """Defines the single training step.
    
    Args:
        state: Current `TrainState` of the model
        batch_data: Tuple containingm a batch of images and their corresponding labels
    Returns:
        loss: Mean loss for the current batch
        accuracy: Mean accuracy for the current batch
        state: Updated state of the model
    """
    
    # 1. Get the images and the labels
    inputs, labels = batch_data

    # 2. Calculate the loss and get the gradients
    (loss, logits), grads = jax.value_and_grad(loss, has_aux=True)(state.params, inputs, labels)
    
    # 3. Calculate the accuracy for the cuurent batch
    accuracy = accuracy(logits, labels)
    
    # 4. Update the state (parameters values) of the model
    state = state.apply_gradients(grads=grads)
    
    # 5. Return loss, accuracy and the updated state
    return loss, accuracy, state

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
    loss, logits = loss(state.params, inputs, labels)
    
    # 3. Calculate the accuracy
    accuracy = accuracy(logits, labels)
    
    # 4. Return loss and accuracy values
    return loss, accuracy


# F. Initial train state including parameters initialization
def create_train_state(rng, lr=1e-4):
    """Creates initial `TrainState for our classifier.
    
    Args:
        key: PRNG key to initialize the model parameters
        lr: Learning rate for the optimizer
    
    """
    # 1. Model instance
    # Initialize the model params and optimizer
    rng = jax.random.PRNGKey(device)
    model = SimpleClassifier(num_hidden=4, num_outputs=2, mean_value=1.0)
    #print(model)

    # 2. Initialize the parameters of the model
    #params = model.init(key, jnp.ones([1, 32, 32, 3]))['params']
    params = model.init(rng, jnp.ones((1, train_data.shape[1])))
    
    # 3. Define the optimizer with the desired learning rate
    optimizer = optax.adam(learning_rate=lr)
    
    # 4. Create and return initial state from the above information. The `Module.apply` applies a 
    # module method to variables and returns output and modified variables.
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)




# Training data
train_data = x_train
train_labels = y_train
validation_data = x_val
validation_labels = y_val

# Test data
# test_data = x_test
# test_labels = y_test

# Training loop
batch_size = 10
num_epochs = 1000
num_batches = train_data.shape[0] // batch_size
validation_interval = 1  # Validate every 10 epochs


# Initialize the model params and optimizer
rng = jax.random.PRNGKey(device)
rng, init_key = jax.random.split(rng)
state = create_train_state(init_key)


training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []

# Training 
for i in range(num_epochs):
    num_train_batches = len(train_data) // batch_size
    num_valid_batches = len(validation_data) // batch_size
    
    # Lists to store loss and accuracy for each batch
    train_batch_loss, train_batch_acc = [], []
    valid_batch_loss, valid_batch_acc = [], []
    
    rng, subkey = jax.random.split(rng)
    indices = jax.random.permutation(subkey, jnp.arange(train_data.shape[0]))

    for batch_idx in range(num_batches):
        
        #step = epoch * decay_steps + batch_idx
        #print('step:', step)
        #current_learning_rate = learning_rate_decay(initial_learning_rate, decay_rate, step)
        
        # Update the optimizer with the current learning rate
        #optimizer = optax.sgd(learning_rate=current_learning_rate, momentum=0.6)
        
        batch_indices = indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch = (train_data[batch_indices], train_labels[batch_indices])
        
        loss_value, acc, state = train_step(state, batch)
        train_batch_loss.append(loss_value)
        train_batch_acc.append(acc)
 
        loss_value, acc = test_step(state, (validation_data, validation_labels))
        valid_batch_loss.append(loss_value)
        valid_batch_acc.append(acc)
        
        # Loss for the current epoch
        epoch_train_loss = jnp.mean(train_batch_loss)
        epoch_valid_loss = jnp.mean(valid_batch_loss)
        
        # Accuracy for the current epoch
        epoch_train_acc = jnp.mean(train_batch_acc)
        epoch_valid_acc = jnp.mean(valid_batch_acc)
        
        training_loss.append(epoch_train_loss)
        training_accuracy.append(epoch_train_acc)
        validation_loss.append(epoch_valid_loss)
        validation_accuracy.append(epoch_valid_acc)
        
        print(f"loss: {epoch_train_loss:.3f}   acc: {epoch_train_acc:.3f}  valid_loss: {epoch_valid_loss:.3f}  valid_acc: {epoch_valid_acc:.3f}")




"""
model = SimpleClassifier(num_hidden=4, num_outputs=2, mean_value=1.0)
print(model)

params = model.init(rng, jnp.ones((1, train_data.shape[1])))  #batch_size,train_data.shape[1]
#print(params)

# Flatten the model parameters
#params_flat, params_tree = jax.tree_flatten(params)
def learning_rate_decay(initial_learning_rate, decay_rate, decay_steps):
    return initial_learning_rate / (1 + (decay_rate*decay_steps))

initial_learning_rate = 0.01
decay_rate = 0.09
decay_steps = train_data.shape[0]/batch_size

optimizer = optax.sgd(learning_rate=initial_learning_rate, momentum=0.5)
#opt_state = optimizer.init(params["params"])
opt_state = optimizer.init(params)

"""

"""for epoch in range(num_epochs):
    rng, subkey = jax.random.split(rng)
    indices = jax.random.permutation(subkey, jnp.arange(train_data.shape[0]))

    for batch_idx in range(num_batches):
        
        #step = epoch * decay_steps + batch_idx
        #print('step:', step)
        #current_learning_rate = learning_rate_decay(initial_learning_rate, decay_rate, step)
        
        # Update the optimizer with the current learning rate
        #optimizer = optax.sgd(learning_rate=current_learning_rate, momentum=0.6)
        
        batch_indices = indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch = (train_data[batch_indices], train_labels[batch_indices])

        #params, opt_state = update(params, opt_state, batch)
        
    # Calculate and print training loss and accuracy at the end of each epoch
    #train_loss = loss(params, (train_data, train_labels))
    #train_accuracy = accuracy(params, (train_data, train_labels))
    train_loss, train_accuracy, state = train_step(state, (train_data, train_labels))
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}, Training Accuracy: {train_accuracy:.2f}")
    
    # Validation  
    if (epoch + 1) % validation_interval == 0:
        #validation_loss = loss(params, (validation_data, validation_labels))
        #validation_accuracy = accuracy(params, (validation_data, validation_labels))
        validation_loss, validation_accuracy, state = train_step(state, (validation_data, validation_labels))
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy:.2f}")"""

"""flat_params, _ = jax.tree_util.tree_flatten(params)
for i, value in enumerate(flat_params):
    print(f"Parameter {i + 1}, Shape: {value.shape}")"""


# After training, you can use the trained model for predictions
# For example, predictions = model.apply({'params': params}, input_data)





