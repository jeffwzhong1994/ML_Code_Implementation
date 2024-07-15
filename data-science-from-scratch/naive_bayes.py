from typing import Set
import re

def tokenize(text: str) -> Set[str]:
	text = text.lower()
	all_words = re.findall("[a-z0-9']+", text)
	return set(all_words)

assert tokenize("Data science is science") == {'data', 'is', 'science'}

from typing import NamedTuple

class Message(NamedTuple):
	text: str
	is_spam: bool

from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:
	def __init__(self, k: float = 0.5) -> None:
		self.k = k

		self.tokens: Set[str] = set()
		self.token_spam_counts: Dict[str, int] = defaultdict(int)
		self.token_ham_counts: Dict[str, int] = defaultdict(int)
		self.spam_messages = self.ham_messages = 0

	def train(self, messages: Iterable[Message]) -> None:
		for message in messages:
			# Increment message counts:
			if message.is_spam:
				self.spam_messages += 1
			else:
				self.ham_messages += 1

			for token in tokenize(message.text):
				self.tokens.add(token)
				if message.is_spam:
					self.token_spam_counts[token] += 1
				else:
					self.token_ham_counts[token] += 1

	def _probabilities(self, token: str) -> Tuple[float, float]:
		"""returns P(token | spam) and P(token | not spam)"""
		spam = self.token_spam_counts[token]
		ham = self.token_ham_counts[token]

		p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
		p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

		return p_token_spam, p_token_ham

	def predict(self, text: str) -> float:
		text_tokens = tokenize(text)
		log_prob_if_spam = log_prob_if_ham = 0.0

		for token in self.tokens:
			prob_if_spam, prob_if_ham = self._probabilities(token)
			
			# If token appear in the message, 
			# add the log probability of seeing it:
			if token in text_tokens:
				log_prob_if_spam += math.log(prob_if_spam)
				log_prob_if_ham += math.log(prob_if_ham)

			#else. add the log prob of not seeing it
			else:
				log_prob_if_spam += math.log(1.0 - prob_if_spam)
				log_prob_if_ham += math.log(1.0 - prob_if_ham)

		prob_if_spam = math.exp(log_prob_if_spam)
		prob_if_ham = math.exp(log_prob_if_ham)
		return prob_if_spam / (prob_if_spam + prob_if_ham)

messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]

model = NaiveBayesClassifier(k=0.5)
model.train(messages)

assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

# Let's make a prediction on a test example:
text = "hello spam"

#print(model.predict(text)) ##0.835

probs_if_spam = [
    (1 + 0.5) / (1 + 2 * 0.5),      # "spam"  (present)
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
    (0 + 0.5) / (1 + 2 * 0.5)       # "hello" (present)
]

probs_if_ham = [
    (0 + 0.5) / (2 + 2 * 0.5),      # "spam"  (present)
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
    (1 + 0.5) / (2 + 2 * 0.5),      # "hello" (present)
]

p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

# Should be about 0.83
assert model.predict(text) == p_if_spam / (p_if_spam + p_if_ham)

# Test Case using a somewhat one dataset: SpamAssassin public corpus
from io import BytesIO   # So we can treat bytes as a file.
import requests          # To download the files, which
import tarfile           # are in .tar.bz format.

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
FILES = ["20021010_easy_ham.tar.bz2",
         "20021010_hard_ham.tar.bz2",
         "20021010_spam.tar.bz2"]

# This is where the data will end up,
# in /spam, /easy_ham, and /hard_ham subdirectories.
# Change this to where you want the data.
# OUTPUT_DIR = 'spam_data'

# for filename in FILES:
#     # Use requests to get the file contents at each URL.
#     content = requests.get(f"{BASE_URL}/{filename}").content

#     # Wrap the in-memory bytes so we can use them as a "file."
#     fin = BytesIO(content)

#     # And extract all the files to the specified output dir.
#     with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
#         tf.extractall(OUTPUT_DIR)

import glob, re

path = 'spam_data/*/*'

data: List[Message] = []

#glob.glob returns every filename that matches the wildcard path
for filename in glob.glob(path):
	is_spam = "ham" not in filename

	#data processing:
	with open(filename, errors='ignore') as email_file:
		for line in email_file:
			if line.startswith("Subject:"):
				subject = line.lstrip("Subject: ")
				data.append(Message(subject, is_spam))
				break

import random
from machine_learning_basic import split_data

random.seed(0)
train_messages, test_messages = split_data(data, 0.75)

model = NaiveBayesClassifier()
model.train(train_messages)

from collections import Counter

predictions = [(message, model.predict(message.text)) for message in test_messages]

#Assume the spam_prob > 0.5 as spam prediction
confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
						for message, spam_probability in predictions)

print(confusion_matrix)


#what words are most or least indicative of spam:
def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
	prob_if_spam, prob_if_ham = model._probabilities(token)

	return prob_if_spam / (prob_if_spam + prob_if_ham)

words = sorted(model.tokens, key = lambda t: p_spam_given_token(t, model))

print("most spam words: ", words[-10:])
print("least spam words: ", words[:10])

#stemmer function:
def drop_final_s(word):
	return re.sub("s$", "", word)