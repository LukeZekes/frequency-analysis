import numpy as np
import matplotlib.pyplot as plt

data = np.load("test_data_0_19.npy")
labels = np.load("test_labels_0_19.npy")
maxGraphs = 6
fig, ax = plt.subplots()
genreIDs = [2, 10, 12, 15, 17, 21, 1235]
# 2D array sorting the indices of the tracks into their respective genres
labelGroups = [[], [], [], [], [], [], []]

for i in range(len(labels)):
    ind = genreIDs.index(labels[i])
    labelGroups[ind].append(i)

# Graph the data from every track in the testing set
for i in range(0, len(labelGroups)):
    for j in labelGroups[i][:min(maxGraphs, len(labelGroups))]:
        ax.plot(data[j])

plt.show()
