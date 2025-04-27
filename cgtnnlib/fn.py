# A treasure trove of functional utilities

def compose(f, g):
    def wrapper(*args, **kwargs):
        return f(g(*args, **kwargs))
    return wrapper