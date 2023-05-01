import os
import fma.utils as utils
from freqAnalysis import ScoreSample
from math import floor


def LoadData(dir, numBins, percentFiles, fromStart=True):
    samplesDir = dir
    sampleGenres = []
    scores = []
    tracks = utils.load("./fma_metadata/tracks.csv")
    genres = utils.load("./fma_metadata/genres.csv")
    # Get the metadata for only the 'small' dataset
    small = tracks[tracks["set", "subset"] <= "small"]
    genreMd = small["track"]["genre_top"]

    # For each track, get its score and genre
    files = os.listdir(samplesDir)
    if fromStart:
        for i in range(floor(len(files) * percentFiles / 100)):
            track = files[i]
            path = os.path.join(samplesDir, track)
            id = track.removesuffix(".mp3")
            # Find the id for its top genre
            genre = genreMd[int(id)]
            genreID = genres.loc[genres["title"] == genre].index[0]

            # Filter out experimental and unclassifiable music
            if genreID != 38 and genreID != 76 and genreID != 125:
                sampleGenres.append(genreID - 1)
                scores.append(ScoreSample(path, numBins))
    else:
        for i in range(len(files) - floor(len(files) * percentFiles / 100), len(files)):
            track = files[i]
            path = os.path.join(samplesDir, track)
            id = track.removesuffix(".mp3")
            # Find the id for its top genre
            genre = genreMd[int(id)]
            genreID = genres.loc[genres["title"] == genre].index[0]

            # Filter out experimental and unclassifiable music
            if genreID != 38 and genreID != 76 and genreID != 125:
                sampleGenres.append(genreID)
                scores.append(ScoreSample(path, numBins))

    return scores, sampleGenres
