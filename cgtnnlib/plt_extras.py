## matplotlib.pyplot extensions v.0.1
## Created at Wed 4 Dec
## v.0.1 set_xlabel, set_ylabel, set_title

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