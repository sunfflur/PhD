import jax
import jax.numpy as jnp
from normalization import NormalizeData
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
seed=11

def splitting(data1, labels1, data2, labels2):
  # put both classes together
  dht_data = jnp.array(NormalizeData(jnp.concatenate((data1, data2))))
  data_labels = jnp.concatenate((labels1, labels2))

  # train trials
  dht_train = dht_data.at[:,:,1:].get()
  print(dht_train.shape)
  x_train = jnp.reshape(dht_train, (dht_train.shape[0]*dht_train.shape[2], dht_train.shape[1]))
  labels_train = data_labels.at[:,1:].get()
  y_train = jnp.reshape(labels_train, (labels_train.shape[0]*labels_train.shape[1], 1))
  y_train = to_categorical(jnp.where(y_train == 15 , 1, 0))

  # test trial
  dht_test = dht_data.at[:,:,0].get()
  x_test = dht_test
  labels_test = data_labels.at[:,0].get()
  y_test = jnp.reshape(labels_test, (labels_test.shape[0], 1))
  y_test = to_categorical(jnp.where(y_test == 15 , 1, 0))
  return x_train, y_train, x_test, y_test



def splitting_v2(data1, labels1, data2, labels2):
  # put both classes together
  dht_data = jnp.array(NormalizeData(jnp.concatenate((data1, data2))))
  data_labels = jnp.concatenate((labels1, labels2))

  X = jnp.reshape(dht_data, (dht_data.shape[0]*dht_data.shape[2], dht_data.shape[1]))
  y = jnp.reshape(data_labels, (data_labels.shape[0]*data_labels.shape[1], 1))
  y = jnp.where(y == 15, 1, 0)

  print(X.shape, y.shape)

  x_train, x_test, y_train, y_test  = train_test_split(X, y, test_size=0.30, random_state=seed, shuffle=True, stratify=y)

  return tf.stack(x_train), tf.stack(x_test), to_categorical(y_train), to_categorical(y_test)

def splitting(data, labels, test_size):
  X = data.reshape((data.shape[0]*data.shape[2], data.shape[1]))
  y = labels.reshape((data.shape[0]*data.shape[1]))
  
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=y)
  return x_train, x_test, to_categorical(y_train), to_categorical(y_test)


def shuffling(x_train, y_train, x_test, y_test):
  idx_train = jax.random.permutation(jax.random.PRNGKey(seed), x_train.shape[0])
  idx_test = jax.random.permutation(jax.random.PRNGKey(seed), x_test.shape[0])
  return tf.stack(x_train[idx_train]), tf.stack(y_train[idx_train]), tf.stack(x_test[idx_test]), tf.stack(y_test[idx_test])