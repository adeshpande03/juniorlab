from dataAnalysis import *


def generateImagesAndData():
    results = [
        (analyzeData("day3_liquidtrial1")),
        (analyzeData("day3_liquidtrial2")),
        (analyzeData("day3_liquidtrial3")),
        (analyzeData("day3_liquidtrial4")),
        (analyzeData("day3_liquidtrial5")),
        (analyzeData("day3_liquidtrial6")),
    ]
    averageHalfLife = sum([i['halfLife'] for i in results])/len(results)
    averageError = math.sqrt(sum([i['error'] ** 2 for i in results])) / math.sqrt(len(results))
    # print(averageHalfLife, averageError)
    return averageHalfLife, averageError


print(f"Results: {generateImagesAndData()}")
