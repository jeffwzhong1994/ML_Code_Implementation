import random
from typing import TypeVar, List, Tuple

X = TypeVar('X')

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
	"""Split data into fractions [prob , 1-prob] """
	data = data[:]
	random.shuffle(data)
	cut = int(len(data) * prob)
	return data[:cut], data[cut:]

data = [n for n in range(1000)]
train, test = split_data(data, 0.75)

Y = TypeVar('Y')

def train_test_split(xs: List[X],
					ys: List[Y],
					test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:

	# generate the indices and split them:
	idxs = [i for i in range(len(xs))]
	train_idxs, test_idxs = split_data(idxs, 1- test_pct)

	return([xs[i] for i in train_idxs],
		[xs[i] for i in test_idxs],
		[ys[i] for i in train_idxs],
		[ys[i] for i in test_idxs])

xs = [x for x in range(1000)]
ys = [2 * x for x in xs]
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
	correct = tp + tn
	total = tp + fp + fn + tn
	return correct / total

def precision(tp: int, fp: int, fn: int, tn: int) -> float:
	return tp / (tp + fp)

def recall(tp: int, fp: int, fn: int, tn: int) -> float:
	return tp / (tp + fn)

def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
	p = precision(tp, fp, fn, tn)
	r = recall(tp, fp, fn, tn)

	return 2 * p * r / (p + r)
	