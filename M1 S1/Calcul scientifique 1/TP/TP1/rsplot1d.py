
from matplotlib.pyplot import *
#from math import *

with open("rsplot1d.dat", "r") as f:
    contenu = f.read().split()

#print(contenu)
np = len(contenu)//3
#print(np)

x = [float(contenu[3*i]) for i in range(np)]
y = [float(contenu[3*i+1]) for i in range(np)]
z = [float(contenu[3*i+2]) for i in range(np)]
plot(x, y, color="blue")
plot(x, z, color="red")
legend(["1", "2"])
show()

