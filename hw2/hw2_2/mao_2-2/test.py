import pickle

with open('data/i2w.pickle', 'rb') as handle:
	a = pickle.load(handle)
with open('data/w2i.pickle', 'rb') as handle:
	b = pickle.load(handle)
for i in range(10):
	print(a[i])
	print(b[a[i]])