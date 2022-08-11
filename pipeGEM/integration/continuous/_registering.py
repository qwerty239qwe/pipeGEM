import inspect


constraint_dict = {"default": (None, []), "None": (None, [])}


def register(func):
    constraint_dict[func.__name__] = (func, list(inspect.signature(func).parameters))
    return func