"""Pluggable samplers: a small registry + a generic ``fit`` dispatcher.

Add a new sampler by subclassing :class:`BaseSampler` and calling :func:`register_sampler`.
"""
from __future__ import annotations

from .abc import ABCSampler, fit_ABC
from .base import BaseSampler, SamplerResult, summarize_posterior

_SAMPLERS = {"abc": ABCSampler}


def register_sampler(name, sampler_cls, *, overwrite=False):
    if name in _SAMPLERS and not overwrite:
        raise ValueError(f"Sampler {name!r} already registered (pass overwrite=True).")
    _SAMPLERS[name] = sampler_cls


def get_sampler(name):
    if name not in _SAMPLERS:
        raise KeyError(f"Unknown sampler {name!r}. Available: {list_samplers()}")
    return _SAMPLERS[name]()


def list_samplers():
    return sorted(_SAMPLERS)


def fit(lc, model, sampler="abc", **kwargs):
    """Generic dispatch: ``fit(lc, model, sampler='abc', ...)``."""
    return get_sampler(sampler).fit(lc, model, **kwargs)


__all__ = [
    "BaseSampler", "SamplerResult", "summarize_posterior",
    "ABCSampler", "fit_ABC", "fit",
    "register_sampler", "get_sampler", "list_samplers",
]
