import librosa
import numpy as np


def ScoreFrequency(y, numBins):
    D = librosa.stft(y, center=False, n_fft=2 * (numBins - 1))
    # Want to get the frequency bin with the largest magnitude at a certain frame t
    T = np.swapaxes(D, 0, 1)
    numFrames = T.shape[0]
    numBins = T.shape[1]
    binsCount = np.zeros(numBins, dtype=int)
    for t in range(0, numFrames):
        # Find the frequency bin with the greatest magnitude at time t
        greatestBin = 0  # Index of the bin with the greatest magnitude at time t
        for f in range(0, numBins):
            mag = abs(T[..., t, f])
            if abs(T[..., t, greatestBin]) < mag:
                greatestBin = f

        # Increment a counter for the greatest bin
        binsCount[greatestBin] = binsCount[greatestBin] + 1

    return binsCount, numFrames


# Returns an array with numBins elements, where the value of an element is the percentage of frames in which the corresponding frequency bin had the largest maginitude of all frequency bins
def ScoreSample(path, numBins, duration=None, offset=0):
    duration = 4
    sr = 22050
    sample = librosa.load(path, mono=True, sr=sr, duration=duration, offset=offset)[0]
    score, numFrames = ScoreFrequency(sample, numBins)
    score = score / numFrames
    return score
