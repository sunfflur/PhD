import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from main import x_train, x_val, x_test, y_train, y_val, y_test # data
import subprocess
from flax.training import train_state


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
        
        w = self.param('weights', nn.initializers.normal(stddev=0.01), (x.shape[1],))
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
        #x = nn.log_softmax(x) # not necessary here...
        return x

# Define loss acc and update functions

def loss(params, batch):
    inputs, labels = batch
    logits = model.apply(params, inputs) #model instead
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
    return loss


def accuracy(params, batch):
    inputs, targets = batch
    logits = model.apply(params, inputs)
    return jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(targets, -1))

# Define a function for updating parameters using the optimizer
@jax.jit
def update(params, opt_state, batch):
    grads = jax.grad(loss)(params, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

# Define inverse time decay schedule - not using yet
def learning_rate_decay(initial_learning_rate, decay_rate, decay_steps):
    return initial_learning_rate / (1 + (decay_rate*decay_steps))

# Training data
train_data = x_train
train_labels = y_train
validation_data = x_val
validation_labels = y_val

# Initialize the model params and optimizer
rng = jax.random.PRNGKey(device)
rng, init_key = jax.random.split(rng)

model = SimpleClassifier(num_hidden=8, num_outputs=2, mean_value=2.0)
#print(model)

params = model.init(rng, jnp.ones((1, train_data.shape[1]))) # how do I init here
#print(params)

batch_size = 10
num_epochs = 100
num_batches = train_data.shape[0] // batch_size
validation_interval = 1  # Validate every N epochs

initial_learning_rate = 0.01
decay_rate = 0.09
decay_steps = train_data.shape[0]/batch_size

optimizer = optax.sgd(learning_rate=initial_learning_rate, momentum=0.6)
opt_state = optimizer.init(params)

# Lists to record loss and accuracy for each epoch
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []

for epoch in range(num_epochs):
    rng, subkey = jax.random.split(rng)
    indices = jax.random.permutation(subkey, jnp.arange(train_data.shape[0]))
    
    # Lists to store loss and accuracy for each batch
    train_batch_loss, train_batch_acc = [], []
    valid_batch_loss, valid_batch_acc = [], []

    for batch_idx in range(num_batches):
        
        step = epoch * decay_steps + batch_idx

        batch_indices = indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch = (train_data[batch_indices], train_labels[batch_indices])

        # Calculate training loss and accuracy per batch
        train_loss = loss(params, batch)
        train_accuracy = accuracy(params, batch)
        train_batch_loss.append(train_loss)
        train_batch_acc.append(train_accuracy)

        params, opt_state = update(params, opt_state, batch) # calls update here
    
    # Loss and acc for the current epoch
    epoch_train_loss = jnp.mean(jnp.asarray(train_batch_loss))
    epoch_train_acc = jnp.mean(jnp.asarray(train_batch_acc))
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_train_loss}, Training Accuracy: {epoch_train_acc:.2f}")
    
    # Validation at the end of each epoch
    if (epoch + 1) % validation_interval == 0:
        epoch_val_loss = loss(params, (validation_data, validation_labels))
        epoch_val_accuracy = accuracy(params, (validation_data, validation_labels))
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {epoch_val_loss}, Validation Accuracy: {epoch_val_accuracy:.2f}")
    
    training_loss.append(epoch_train_loss)
    training_accuracy.append(epoch_train_acc)
    validation_loss.append(epoch_val_loss)
    validation_accuracy.append(epoch_val_accuracy)