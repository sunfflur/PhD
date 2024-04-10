import jax
import jax.numpy as jnp
from experiments.main_functions.utils import to_categorical, shuffling
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
seed=0
  
def splitting(data, labels, test_size, val_size, n_classes):
  
  X = jnp.reshape(data, (data.shape[0]*data.shape[2], data.shape[1]))
  y = jnp.reshape(labels, (labels.shape[0]*labels.shape[1]))
  
  #X = data.reshape((data.shape[0]*data.shape[2], data.shape[1]))
  #y = labels.reshape((labels.shape[0]*labels.shape[1]))
  
  xx_train, x_test, yy_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=y)
  x_train, x_val, y_train, y_val = train_test_split(xx_train, yy_train, test_size=val_size, random_state=seed, shuffle=True, stratify=yy_train)
  
  return x_train, x_val, x_test, to_categorical(y_train, n_classes=n_classes), to_categorical(y_val, n_classes=n_classes), to_categorical(y_test, n_classes=n_classes)



def splitting_per_trial(data, labels, test_trial, val_size, n_classes):
  
  # create mask based on the test_trial (~mask == not mask)

  mask = jnp.arange(6) == test_trial
  # Split the data into test and train sets
  test_set = data[:, :, mask]
  test_labels = labels[:, mask]
  train_set = data[:, :, ~mask]
  train_labels = labels[:, ~mask]

  # reshape to take off the last dimension
  test_data = jnp.reshape(test_set, (test_set.shape[0]*test_set.shape[2], test_set.shape[1]))
  test_ydata = jnp.reshape(test_labels, (test_labels.shape[0]*test_labels.shape[1]))

  train_data = jnp.reshape(train_set, (train_set.shape[0]*train_set.shape[2], train_set.shape[1]))
  train_ydata = jnp.reshape(train_labels, (train_labels.shape[0]*train_labels.shape[1]))

  # Split the validation and training sets
  x_train, x_val, y_train, y_val = train_test_split(train_data, 
                                                    train_ydata, 
                                                    test_size=val_size, 
                                                    random_state=seed, 
                                                    shuffle=True, 
                                                    stratify=train_ydata)
  
  x_test, y_test = shuffling(test_data, test_ydata)
  
  return x_train, x_val, x_test, to_categorical(y_train, n_classes=n_classes), to_categorical(y_val, n_classes=n_classes), to_categorical(y_test, n_classes=n_classes)