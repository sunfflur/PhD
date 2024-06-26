import jax
import jax.numpy as jnp
from main_functions.utils import to_categorical, shuffling
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

seed=0
  
def splitting(data, labels, test_size, val_size, n_classes):
  
  # its working for data shape (64, 1000, 6) not for (16, 4, 1000, 6)
  X = jnp.reshape(data, (data.shape[0]*data.shape[2], data.shape[1]))
  y = jnp.reshape(labels, (labels.shape[0]*labels.shape[1]))
  
  #X = data.reshape((data.shape[0]*data.shape[2], data.shape[1]))
  #y = labels.reshape((labels.shape[0]*labels.shape[1]))
  
  xx_train, x_test, yy_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=y)
  x_train, x_val, y_train, y_val = train_test_split(xx_train, yy_train, test_size=val_size, random_state=seed, shuffle=True, stratify=yy_train)
  
  return x_train, x_val, x_test, to_categorical(y_train, n_classes=n_classes), to_categorical(y_val, n_classes=n_classes), to_categorical(y_test, n_classes=n_classes)


def splitting_per_trial(data, labels, split_train, split_val, split_test, trial_gab, n_classes):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_val = []
    y_val = []
    all_idxs = jnp.array(trial_gab) #gabarito dos trials por linha
    for trial in split_train:
      x_train.append(data[all_idxs==trial]) #appendo os dados onde bate o valor com o do trial do split
      y_train.append(labels[all_idxs==trial]) #msm coisa
    for trial in split_test:
      x_test.append(data[all_idxs==trial])
      y_test.append(labels[all_idxs==trial])
    for trial in split_val:
      x_val.append(data[all_idxs==trial])
      y_val.append(labels[all_idxs==trial])
      

    if x_val == []:
      return jnp.concatenate(x_train, axis=0),jnp.concatenate(x_test, axis=0), to_categorical(jnp.concatenate(y_train, axis=0), n_classes=n_classes), to_categorical(jnp.concatenate(y_test, axis=0), n_classes=n_classes)
    else:
      return jnp.concatenate(x_train, axis=0), jnp.concatenate(x_val, axis=0), jnp.concatenate(x_test, axis=0), to_categorical(jnp.concatenate(y_train, axis=0), n_classes=n_classes), to_categorical(jnp.concatenate(y_val, axis=0),n_classes=n_classes), to_categorical(jnp.concatenate(y_test, axis=0), n_classes=n_classes)
    
    
def splitting_bcic2a(data, labels, n_trials, val_trial):
  #data has shape 288, 22*1250
  # 288/6 = 48
  trial_size = data.shape[0]//n_trials
  
  start_index = val_trial * trial_size
  end_index = start_index + trial_size
  x_val = data[start_index:end_index, :]
  y_val = labels[start_index:end_index]
  x_train_start = data[:start_index,:]
  y_train_start = labels[:start_index]
  x_train_stop = data[end_index:,:]
  y_train_stop = labels[end_index:]
  x_train = jnp.concatenate((x_train_start, x_train_stop), axis=0)
  y_train = jnp.concatenate((y_train_start, y_train_stop), axis=0)
  return x_train, x_val, y_train, y_val