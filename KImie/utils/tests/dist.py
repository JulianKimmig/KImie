from __future__ import annotations
from typing import Literal, Tuple, List, Type, Any, Dict, Iterable
import numpy as np
from scipy.stats import norm


def wald_wolfowitz_runs_test(s1: Iterable, s2: Iterable):
    """
    Wald-Wolfowitz runs test.
    Tests the null hypothesis that the elements of s2 are randomly distributed in s1.
    If the p-value is less than your chosen significance level (e.g., 0.05), you would reject the null hypothesis.
    :param s1: list of elements
    :param s2: list of elements
    :return: z-score and p-value
    """
    # Create a binary sequence: 1 if the element is in s2, 0 otherwise
    binary_sequence = [int(x in s2) for x in s1]

    # Count the number of runs
    runs = np.array(binary_sequence[:-1]) != np.array(binary_sequence[1:])
    num_runs = np.count_nonzero(runs) + 1

    # Calculate the expected number of runs and the standard deviation
    n1 = np.sum(binary_sequence)
    n2 = len(s1) - n1
    expected_runs = 2.0 * n1 * n2 / len(s1) + 1
    runs_std_dev = np.sqrt((expected_runs - 1) * (expected_runs - 2) / (len(s1) - 1))

    # Calculate the z-score
    z = (num_runs - expected_runs) / runs_std_dev

    # Calculate the two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return z, p_value
