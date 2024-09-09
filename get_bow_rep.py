


import string
import csv
import pandas as pd

df = pd.read_csv('multi_vocab')
print("File read successfully.")

# csv_header = ['sentiment_label']
# indices = dict()
# indices['sentiment_label'] = 0

# i = 1
# for l in open('multi_vocab'):
# 	csv_header.append(l.strip())
# 	indices[l.strip()] = i
# 	i += 1

# reader = csv.DictReader(open('IMDB_Dataset.csv'))

# df = pd.DataFrame(columns=csv_header)

# i = 0
# for row in reader:
# 	if i % 100 == 0:
# 		print(i)
# 	newrow = [0]*len(csv_header)
# 	if row['sentiment'].strip() == 'positive':
# 		newrow[0] = 1

# 	review = row['review']

# 	for w in review.split():
# 		if w.lower() in csv_header:
# 			token = w.lower()
# 			newrow[indices[token]] += 1

# 		elif w.lower().strip(string.punctuation) in csv_header:
# 			token = w.lower().strip(string.punctuation)
# 			newrow[indices[token]] += 1

# 	if i == 0:
# 		print(newrow[:10])

# 	df.loc[len(df.index)] = newrow
# 	i += 1
# df.to_pickle('IMDB_BOW.pkl')
# df.to_csv('IMDB_BOW.csv')



