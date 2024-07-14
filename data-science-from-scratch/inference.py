from typing import Tuple
import math

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
	""" Returns mu and sigma corresponding to a Binomial(n, p)"""
	mu = p * n
	sigma = math.sqrt(p * (1 - p) * n)
	return mu, sigma

from probability import normal_cdf

# The normal cdf is the probability the variable is below a threshold:
normal_probability_below = normal_cdf

# It's above the threshold if its not below the threshold:
def normal_probability_above(lo: float,
							mu: float = 0,
							sigma: float = 1) -> float:

	""" The probability that a N(mu, sigma) is greater than lo. """
	return 1 - normal_cdf(lo, mu, sigma)

# It's between if it's less than hi, but no less than lo
def normal_probability_between(lo: float,
							hi: float,
							mu: float = 0,
							sigma: float = 1) -> float:
	return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# It's outside if it's not between:
def normal_probability_outside(lo: float,
							hi: float,
							mu: float = 0,
							sigma: float = 1) -> float:

	return 1 - normal_probability_between(lo, hi, mu, sigma)

# Code the inverse normal cdf:
from probability import inverse_normal_cdf
from typing import Tuple

def normal_upper_bound(probability: float,
					mu: float = 0,
					sigma: float = 1) -> float:
	"""Returns the z for which P(Z <= z) = probability """
	return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
					mu: float = 0,
					sigma: float = 1) -> float:
	"""Returns the z for which P(Z >= z) = probability """
	return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float,
							mu: float = 0,
							sigma: float = 1) -> Tuple[float, float]:
	"""
	Returns the symmetric (about the means) bounds
	that contains certain probability
	"""
	tail_probability = (1 - probability) / 2
	upper_bound = normal_upper_bound(tail_probability, mu, sigma)
	lower_bound = normal_lower_bound(tail_probability, mu, sigma)

	return lower_bound, upper_bound

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
print("mu:", mu_0, "sigma:", sigma_0)

lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)
print("lower_bound:", lo, "upper_bound:", hi, "\n")

# Let's delve into the power of the experiment a bit more:

## Assume the actual mu and sigma is based on p = 0.55
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# calculate the power of the experiment:
type_2_probability = abs(normal_probability_between(lo, hi, mu_1, sigma_1))
power = 1 - type_2_probability
print("power of the experiment is: ", power)

#how about power of one-sided test?
hi = normal_upper_bound(0.95, mu_0, sigma_0)
type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability
print("power of one-sided experiment is: ", power)

# Onto p-value of the calculations:
def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
	"""
	How likely are we to see a value at least as extreme as x 
	(in either direction) if our values are from a N(mu,sigma)?
	"""
	if x >= mu:
		return 2 * normal_probability_above(x, mu, sigma)
	else:
		return 2 * normal_probability_below(x, mu, sigma)

print("two sided p-value of 529.5 is:", two_sided_p_value(529.5, mu_0, sigma_0))

# Convince yourself whether the p-value is true with a simulation:
import random

extreme_value_count = 0
for _ in range(1000):
	num_heads = sum(1 if random.random() < 0.5 else 0 for _ in range(1000))
	if num_heads >= 530 or num_heads <= 470:
		extreme_value_count += 1

print("extreme value count in 1000 simulations:", extreme_value_count)

# Plot Confidence Interval:
p_hat = 525/ 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1-p_hat) / 1000)
print("confidence interval is: ", normal_two_sided_bounds(0.95, mu, sigma))

# p-Hacking:
from typing import List

def run_experiment() -> List[bool]:
	"""Flips a fair coin 1000 times, Head= True, Tail = False"""
	return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment: List[bool]) -> bool:
	""" 5% significance level"""
	num_heads = len([flip for flip in experiment if flip])
	return num_heads < 469 or num_heads > 531

random.seed(0)
experiments = [run_experiment() for _ in range(1000)]
num_rejects = len(
				[experiment for experiment in experiments
				if reject_fairness(experiment)]
				)
print("number of rejections is: ", num_rejects)

# A/B testing:
def estimated_parameters(N:int, n:int) -> Tuple[float, float]:
	p = n / N
	sigma = math.sqrt(p * (1-p) / N)
	return p, sigma

def a_b_test_statistics(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
	p_A, sigma_A = estimated_parameters(N_A, n_A)
	p_B, sigma_B = estimated_parameters(N_B, n_B)

	return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)
