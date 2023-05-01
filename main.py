import matplotlib.pyplot as plt
import numpy as np
from labelData import LoadData
import os
import tensorflow as tf


def TrainRange(start, end, initData, initLabels, fromStart=True):
    # Get the data for all tracks in a directory
    sampleDirs = [d for d in os.scandir("./samples/fma_small") if d.is_dir()]
    trainData = np.load(initData)
    trainLabels = np.load(initLabels)
    for i in range(start, end):
        dir = sampleDirs[i]
        dirPath = dir.path
        data, labels = LoadData(
            dirPath, 40, 80 if fromStart else 20, fromStart=fromStart)
        if len(data) != 0:
            data = np.asarray(data)
            labels = np.asarray(labels)
            trainData = np.concatenate((trainData, data))
            trainLabels = np.concatenate((trainLabels, labels))

    return trainData, trainLabels


def MapLabels(labels):
    for i in range(len(labels)):
        v = labels[i]
        if v == 2:
            labels[i] = 0
        elif v == 10:
            labels[i] = 1
        elif v == 12:
            labels[i] = 2
        elif v == 15:
            labels[i] = 3
        elif v == 17:
            labels[i] = 4
        elif v == 21:
            labels[i] = 5
        elif v == 1235:
            labels[i] = 6

    return labels


if not os.path.exists("./trained_model_7"):
    # Create a new model
    # Get the data for all tracks in a directory
    sampleDirs = [d for d in os.scandir("./samples/fma_small") if d.is_dir()]
    trainData = np.load("train_data_0_19.npy")
    trainLabels = np.load("train_labels_0_19.npy")
    trainLabels = MapLabels(trainLabels)
    # Build a model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(40,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(7, activation="softmax"),
        ]
    )
    # Compile the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    # Train the model
    model.fit(trainData, trainLabels, epochs=1125)

    # Save the model
    model.save("trained_model_7")

else:
    model = tf.keras.models.load_model("./trained_model_6")
# Test the model

if not os.path.exists("./test_data_0_19.npy"):
    testData, testLabels = TrainRange(
        1, 20, "init_test_data.npy", "init_test_labels.npy", False)
else:
    testData = np.load("test_data_0_19.npy")
    testLabels = np.load("test_labels_0_19.npy")

testLabels = MapLabels(testLabels)
correctPredictions = 0
for i in range(len(testData)):
    t = np.expand_dims(testData[i], axis=0)
    p = np.argmax(model.predict(t))
    if p == testLabels[i]:
        correctPredictions += 1

print("% Correct predictions: ", (correctPredictions * 100 / len(testData)))
# fig, ax = plt.subplots()
# colors = ["b", "r", "g", "k", "c", "m"]
# for i in range(0, numSamples):
#     sampleData = scores[i]
#     name = "s" + str(i)
#     bottom = 0
#     for j in range(0, numBins):
#         ax.bar(
#             name, sampleData[j], 0.5, bottom, color="C" + str(j), label="Bin " + str(j)
#         )
#         bottom += sampleData[j]
# plt.show()
