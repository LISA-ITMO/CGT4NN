## matplotlib.pyplot extensions v.0.1
## Created at Wed 4 Dec
## v.0.1 set_xlabel, set_ylabel, set_title

import matplotlib.pyplot as plt

def plot_list(data, x_label="Index", y_label="Value"):
  """
  Plots a list of floats using matplotlib.

  Args:
    data: A list of float values to plot.
    x_label: (Optional) The label for the x-axis. Defaults to "Index".
    y_label: (Optional) The label for the y-axis. Defaults to "Value".
  """

  plt.plot(data)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.grid(True)
  plt.show()

def set_xlabel(ax_or_plt, text):
    is_ax = hasattr(ax_or_plt, 'set_xlabel')
    if is_ax:
        ax_or_plt.set_xlabel(text)
    else:
        ax_or_plt.xlabel(text)


def set_ylabel(ax_or_plt, text):
    is_ax = hasattr(ax_or_plt, 'set_ylabel')
    if is_ax:
        ax_or_plt.set_ylabel(text)
    else:
        ax_or_plt.ylabel(text)


def set_title(ax_or_plt, text):
    is_ax = hasattr(ax_or_plt, 'set_title')
    if is_ax:
        ax_or_plt.set_title(text)
    else:
        ax_or_plt.title(text)