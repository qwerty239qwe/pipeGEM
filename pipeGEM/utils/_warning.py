from functools import wraps, partial
from warnings import warn


CITATIONS = {"gapsplit": {"url": "https://github.com/jensenlab/gapsplit/blob/master/python/gapsplit.py",
                          "doi": "https://doi.org/10.1093/bioinformatics/btz971"},
             "default": {"url": "", "doi": ""}}


def usage_warning(func=None, *, to_cite, usage):
    if func is None:
        return partial(usage_warning, to_cite="default", usage="")

    @wraps(func)
    def print_warning(*args, **kwargs):
        warn(f"The {func.__name__} is used for {usage}. "
             f"Please cite the original source: "
             f"DOI: {CITATIONS[to_cite]['doi']} URL: {CITATIONS[to_cite]['url']}")
        return func(*args, **kwargs)
    return print_warning