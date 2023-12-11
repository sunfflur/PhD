import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from main import x_train, x_test, y_train, y_test  # Assuming these are your data

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
    print(f"Epoch {epoch + 1}, Loss: {loss}")