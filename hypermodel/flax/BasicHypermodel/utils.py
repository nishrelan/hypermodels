from fractions import Fraction
import sys
import jax.numpy as jnp
import matplotlib.pyplot as plt


def omegaconf_list_to_array(l):
    l = [float(Fraction(x)) for x in l]
    return jnp.array(l)


def sftf(s):
    return float(Fraction(s))
