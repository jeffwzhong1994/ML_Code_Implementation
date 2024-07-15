# Using underscores to indicate that these are "private"
# functions, as they are intended to be called by our median
# function but not by other people
from typing import List

def _median_odd(xs: List[float]) -> float:
	""" If len(xs) is odd"""
	return sorted(xs)[len(xs) // 2]

def _median_even(xs: List[float]) -> float:
	""" If len(xs) is even"""
	sorted_xs = sorted(xs)
	hi_midpoint = len(xs) // 2
	return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2

def median(v: List[float]) -> float:
	""" Finds the 'middle-most' value of v """
	return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

def quantile(xs: List[float], p: float) -> float:
	""" Returns the p-th percentile value in x"""
	p_index = int(p*len(xs))
	return sorted(xs)[p_index]

def mode(x: List[float]) -> List[float]:
	counts = Counter(x)
	max_count = max(counts.values())

	return [x_i for x_i, count in counts.item() 
			if count == max_count]

def data_range(xs: List[float]) -> float:
	return max(xs) - min(xs)

from linear_algebra import sum_of_squares

def mean(xs: List[float]) -> float:
	return sum(xs) / len(xs)

def de_mean(xs: List[float]) -> List[float]:
	"""Translate xs by subtracting its mean """
	x_bar = mean(xs)
	return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
	"""Almost the avg. squared deviation from the mean"""
	assert len(xs) >= 2, "variance requires at least two elements"

	n = len(xs)
	deviations = de_mean(xs)
	return sum_of_squares(deviations) / (n - 1)

import math

def standard_deviation(xs: List[float]) -> float:
	return math.sqrt(variance(xs))


def interquartile_range(xs: List[float]) -> float:
	return quantile(xs, 0.75) - quantile(xs, 0.25)

from linear_algebra import dot

def covariance(xs: List[float], ys: List[float]) -> float:
	assert len(xs) == len(ys), "xs and ys must have the same number of elements"

	return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)

def correlation(xs: List[float], ys: List[float]) -> float:
	stdev_x = standard_deviation(xs)
	stdev_y = standard_deviation(ys)
	if stdev_x > 0 and stdev_y > 0:
		return covariance(xs, ys) / stdev_x / stdev_y
	else:
		return 0


