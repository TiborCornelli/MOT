import numpy as np
import matplotlib.pyplot as plt
import csv

down_times = 0.001 * np.array([5,3,7,4,8,10,12,15,17,19, 20, 23, 27, 32, 36, 45, 65, 75, 85])

time_stamps = np.array([
    [0.19, 0.22, 0.3, 0.52, 0.56], # 5ms
    [0.18, 0.2, 0.28, 0.5, 0.54], # 3ms
    [0.17, 0.22, 0.28, 0.5, 0.54], # 7ms
    [0.18, 0.21, 0.28, 0.5, 0.55], # 4ms
    [0.1, 0.14, 0.18, 0.43, 0.5], # 8ms
    [0.1, 0.16, 0.21, 0.43, 0.5], #10ms
    [0.1, 0.15,0.2,0.43,0.51], #12ms
    [0.1, 0.17, 0.21, 0.43,0.5], #15ms
    [0.1, 0.17, 0.21, 0.43, 0.5], #17ms
    [0.1, 0.17, 0.21, 0.43, 0.5], #19ms
    [0.095, 0.17, 0.21, 0.43, 0.5], # 20ms
    [0.085, 0.16, 0.21, 0.43, 0.5], # 23ms
    [0.085, 0.16, 0.21, 0.43, 0.5 ], # 27ms
    [0.085, 0.15, 0.21, 0.43, 0.5], # 32ms
    [0.08, 0.15, 0.21, 0.43, 0.5], # 36ms
    [0.16, 0.25, 0.3, 0.52, 0.58], # 45ms
    [0.13, 0.25, 0.3, 0.52, 0.58], # 65ms
    [0.12, 0.25, 0.3, 0.52, 0.58], # 75ms
    [0.12, 0.25, 0.3, 0.52, 0.58], # 85ms
])

avg_ratios = np.zeros(len(time_stamps))

for i in range(0, len(time_stamps)):
    ratio_sum = 0
    for j_file in range(4):
        file_index = i * 4 + j_file + 1
        with open(f"Data/NewFile{file_index}.csv", newline='') as f:
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

        down_time = down_times[i]


        channel = 2
        ch = data[:, channel - 1]
        t = np.arange(len(ch)) * increment

        mean_left = np.mean(ch[t < time_stamps[i, 0]])
        mean_middle = np.mean(ch[(t > time_stamps[i, 1]) & (t < time_stamps[i, 2])])
        offset = np.mean(ch[(t > time_stamps[i, 3]) & (t < time_stamps[i, 4])])

        N_0 = mean_left - offset
        N_t = mean_middle - offset
        ratio = N_t / N_0
        ratio_sum += ratio
        avg_ratios[i] = ratio_sum / 4

        t_min = 0
        t_max = 1

        index_range = (t > t_min) & (t < t_max) 

        # plt.plot(t[index_range], ch[index_range])
        # plt.xlabel("Zeit [s]")
        # plt.ylabel(f"CH{channel}-Spannung [V]")
        # plt.title(f"File {file_index}, {down_time * 1000} ms, CH{channel}")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

plt.scatter(down_times, avg_ratios)
# Title: I_0 in Latex
plt.title(rf"I_0")
plt.show()
