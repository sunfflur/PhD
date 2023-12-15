import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from main import x_train, x_val, x_test, y_train, y_val, y_test  # Assuming these are your data
import subprocess
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

# Define the neural network model using FLAX
class SimpleClassifier(nn.Module):
    """SimpleClassifier
    Define the neural network model using FLAX
    
    """
    num_hidden: int
    num_outputs: int 

    @nn.compact  # Tells Flax to look for defined submodules
    def __call__(self, x):
        # Perform the calculation of the model to determine the prediction
        # while defining necessary layers
        x = nn.Dense(features=self.num_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_outputs)(x)
        x = nn.log_softmax(x)
        return x


# Initialize the model and optimizer
rng = jax.random.PRNGKey(device)
model = SimpleClassifier(num_hidden=8, num_outputs=4)
# Printing the model shows its attributes
print(model)

params = model.init(rng, jnp.ones((10, 1000)))  
#print(params)

optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(params["params"])

# Define the loss and acc functions
def loss(params, batch):
    inputs, labels = batch
    logits = model.apply(params, inputs)  
    return -jnp.mean(jnp.sum(logits * labels, axis=-1))

def accuracy(params, inputs, targets):

    logits = model.apply(params, inputs)
    predicted_labels = jnp.argmax(logits, axis=-1)
    #print('predicted:', predicted_labels)
    correct_predictions = jnp.sum(predicted_labels == jnp.argmax(targets))
    #print('correct:', correct_predictions)
    total_samples = inputs.shape[0]
    accuracy = correct_predictions / total_samples
    return accuracy.item()

# Define a function for updating parameters using the optimizer
@jax.jit
def update(params, opt_state, batch):
    grads = jax.grad(loss)(params, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

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
num_epochs = 10
num_batches = train_data.shape[0] // batch_size
validation_interval = 10  # Validate every 10 epochs, adjust as needed

for epoch in range(num_epochs):
    rng, subkey = jax.random.split(rng)
    indices = jax.random.permutation(subkey, jnp.arange(train_data.shape[0]))

    for batch_idx in range(num_batches):
        batch_indices = indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch = (train_data[batch_indices], train_labels[batch_indices])

        params, opt_state = update(params, opt_state, batch)

    # Calculate and print training loss and accuracy at the end of each epoch
    train_loss = loss(params, (train_data, train_labels))
    train_accuracy = accuracy(params, train_data, train_labels)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")

    # Validation
    if (epoch + 1) % validation_interval == 0:
        validation_loss = loss(params, (validation_data, validation_labels))
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {validation_loss}")

# After training, you can use the trained model for predictions
# For example, predictions = model.apply({'params': params}, input_data)













"""
# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    hidden_dim: int
    num_classes: int

    def setup(self):
        return [
            nn.Dense(name="dense1", features=self.hidden_dim),
            nn.relu,
            nn.Dense(name="dense2", features=self.hidden_dim),
            nn.relu,
            nn.Dense(name="dense3", features=self.hidden_dim),
            nn.relu,
            nn.Dense(name="dense4", features=self.num_classes)
        ]

    def __call__(self, x):
        for layer in self.setup():
            x = layer(x)
        return x
    
    def init(self, key, x):
        _, params = self.init_with_output(key, x)
        return params

    def apply_with_init(self, key, x):
        return self.apply({'params': self.init(key, x)}, x)

# Define a categorical cross-entropy loss function
def categorical_cross_entropy(predictions, targets):
    return -jnp.sum(targets * jax.nn.log_softmax(predictions))

# Define a training step for categorical cross-entropy
@jax.jit
def train_step_categorical(params, x, y):
    def loss_fn(params, x, y):
        model = SimpleNN(hidden_dim=2, num_classes=num_classes)
        predictions = model.apply({'params': params}, x)
        loss = jnp.mean(categorical_cross_entropy(predictions, y))
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grad = grad_fn(params, x, y)
    updates, _ = optax.sgd(learning_rate=0.01).update(grad, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, loss

# Instantiate the modified model
rng = jax.random.PRNGKey(0)
num_classes = 4  # Change this to your actual number of classes
model = SimpleNN(hidden_dim=2, num_classes=num_classes)

# Use a sample from x_train for initialization
params = model.init(rng, x_train[0:1, :])

# Example usage:
x_data = x_train
y_data = y_train

# Training loop with a specified number of epochs
num_epochs = 10  # Adjust this to the desired number of epochs

for epoch in range(num_epochs):
    params, loss = train_step_categorical(params, x_data, y_data)
    print(f"Epoch {epoch + 1}, Loss: {loss}")"""