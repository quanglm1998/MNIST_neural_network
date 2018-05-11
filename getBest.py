import pickle

fin = open("best.txt", "rb")
a = pickle.load(fin)
a = pickle.load(fin)
b = a.split(", ")
for i in range(len(b)):
	print(b[i])
fin.close()
