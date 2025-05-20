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
    """
    Sets the x-axis label for a matplotlib axes object or figure.

        This function intelligently determines whether the input is an axes
        object (with a `set_xlabel` method) or a figure object (with an
        `xlabel` method) and calls the appropriate method to set the x-axis label.

        Args:
            ax_or_plt: The matplotlib axes object or figure object.
            text: The text for the x-axis label.

        Returns:
            None
    """
    is_ax = hasattr(ax_or_plt, "set_xlabel")
    if is_ax:
        ax_or_plt.set_xlabel(text)
    else:
        ax_or_plt.xlabel(text)


def set_ylabel(ax_or_plt, text):
    """
    Sets the y-axis label for a matplotlib axes object or plot.

        This function intelligently handles both `Axes` objects (which have a
        `set_ylabel` method) and pyplot-style plotting interfaces (which use
        `ylabel`).

        Args:
            ax_or_plt: The Axes object or the pyplot interface to modify.
            text: The text for the y-axis label.

        Returns:
            None
    """
    is_ax = hasattr(ax_or_plt, "set_ylabel")
    if is_ax:
        ax_or_plt.set_ylabel(text)
    else:
        ax_or_plt.ylabel(text)


def set_title(ax_or_plt, text):
    """
    Sets the title of a matplotlib axes object or figure.

        This function intelligently sets the title based on whether the input is an
        axes object (with a `set_title` method) or a figure object (with a `title`
        method).

        Args:
            ax_or_plt: The matplotlib axes or figure object to set the title of.
            text: The text for the title.

        Returns:
            None
    """
    is_ax = hasattr(ax_or_plt, "set_title")
    if is_ax:
        ax_or_plt.set_title(text)
    else:
        ax_or_plt.title(text)
