# A treasure trove of functional utilities


def compose(f, g):
    """
    Composes two functions.

        This function takes two functions, f and g, and returns a new function that
        applies g first, then applies f to the result of g.

        Args:
            f: The outer function.
            g: The inner function.

        Returns:
            A callable representing the composition of f and g.  The returned
            function takes the same arguments as g and returns the result of
            applying f to g's output.
    """

    def wrapper(*args, **kwargs):
        return f(g(*args, **kwargs))

    return wrapper
