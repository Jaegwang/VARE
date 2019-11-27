f = open("Bora.spec", "r")

line = f.readline()

line = f.readline()
version = line.split(": ")[1]
print(version)

f.close()
