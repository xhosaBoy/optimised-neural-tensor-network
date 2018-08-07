# Import TensorFlow and some other libraries we'll be using.
import datetime
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


# plt.ioff()

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# Helper plotting routine.
def display_images(gens, title=""):
  fig, axs = plt.subplots(1, 10, figsize=(25, 3))
  fig.suptitle(title, fontsize=14, fontweight='bold')

  for i in range(10):
    reshaped_img = (gens[i].reshape(28, 28) * 255).astype(np.uint8)
    axs.flat[i].imshow(reshaped_img)
    # axs.flat[i].axis('off')
  return fig, axs


# Helper functions to plot training progress.
def my_plot(list_of_tuples):
  """Take a list of (epoch, value) and split these into lists of
  epoch-only and value-only. Pass these to plot to make sure we
  line up the values at the correct time-steps.
  """
  plt.plot(*zip(*list_of_tuples))


def plot_multi(values_lst_1, labels_lst_1, y_label_1, x_label_1, values_lst_2, labels_lst_2, y_label_2, x_label_2):
  # Plot multiple curves.

  # assert len(values_lst) == len(labels_lst)
  plt.subplot(2, 1, 1)

  for v in values_lst_1:
    my_plot(v)

  plt.legend(labels_lst_1, loc='upper left')

  plt.xlabel(x_label_1)
  plt.ylabel(y_label_1)

  plt.subplot(2, 1, 2)

  for v in values_lst_2:
    my_plot(v)

  plt.legend(labels_lst_2, loc='upper left')

  plt.xlabel(x_label_2)
  plt.ylabel(y_label_2)

  plt.show()
