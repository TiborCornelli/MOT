import numpy as np
import matplotlib.pyplot as plt
import csv

with open(f"Data/NewFile73.csv", newline='') as f:
    reader = csv.reader(f)
    next(reader)
    second_row = next(reader)
    increment = float(second_row[-1])
    next(reader)  

    data = []
    for row in reader:
        values = [float(x) for x in row if x.strip()]
        data.append(values)

data = np.array(data)

channel = 2
ch = data[:, channel - 1]
t = np.arange(len(ch)) * increment

t_min = 0
t_max = 1

index_range = (t > t_min) & (t < t_max) 

plt.plot(t[index_range], ch[index_range])
plt.xlabel("Zeit [s]")
plt.ylabel(f"CH{channel}-Spannung [V]")
plt.title(f"CH{channel}")
plt.grid(True)
plt.tight_layout()
plt.show()
