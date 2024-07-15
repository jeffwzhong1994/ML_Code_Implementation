#stdin and stdout:
#egreg.py
import sys, re

regex = sys.argv[1]

for line in sys.stdin:
	#if it matches the regex, write it to stdout:
	if re.search(regex, line):
		sys.stdout.write(line)

#line_count.py
count = 0
for line in sys.stdin:
	count += 1

print(count)

"""
Usage: cat SomeFile.txt | python egreg.py "[0-9]" | python line_count.py
"""

#most_common_word.py
from collections import Counter

try::
	num_words = int(sys.argv[1])
except:
	print("usage: most_common_word.py num_words")
	sys.exit(1)


count = Counter(word.lower()
				for line in sys.stdin
				for word in line.strip().split()
				if word)

for word, count in counter.most_common(num_words):
	sys.stdout.write(str(count))
	sys.stdout.write("\t")
	sys.stdout.write(word)
	sys.stdout.write("\n")

# Basic of Reading Files:
# 'r' means read-only, 'w' means writes, 'a' means append
file_for_reading = open('reading_file.txt', 'r')

#we could use it within a `with` block so at the end the file will be closed automatically:
with open(filename) as f:
	data = function_that_gets_data(f)

# Process delimited-file:
import csv

with open('tab_delimited_file.txt') as f:
	tab_reader = csv.reader(f, delimiter='\t')
	for row in tab_reader:
		data = row[0]
		symbol = row[1]
		closing_price = float(row[2])
		process(date, symbol, closing_price)

# or get each row as dict (headers as the key)
with open('stock_price.txt') as f:
	colon_reader = csv.DictReader(f, delimiter=':')
	for dict_row in colon_reader:
		date = dict_row['date']
		symbol = dict_row['symbol']
		closing_price = float(dict_row['closing_price'])
		process(date, symbol, closing_price)


import requests, json
github_user = "jeffwzhong1994"
endpoint = f"https://api.github.com/users/{github_user}/repos"

repos = json.loads(requests.get(endpoint).text)

last_5_repos = sorted(repos,
					key = lambda r:r["pushed_at"],
					reverse=True)[:5]

last_5_languages = [repo["language"] for repo in last_5_repos]
