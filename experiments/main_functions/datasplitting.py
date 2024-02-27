import jax
import jax.numpy as jnp
from main_functions.utils import to_categorical
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