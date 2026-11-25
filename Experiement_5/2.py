# For a given population of size 'n'(1000)(generate using random numbers), select 'k'(5)(k<<<n) samples of size 'n'(50)(n'<n) and find the best sample among 'k' samples by measuring goodness(MoCt,Nos) of samples


import random
import math

def mean(data):
    return sum(data) / len(data)


def median(data):
    temp = sorted(data)
    n = len(temp)
    if n % 2 == 0:
        return (temp[n//2 - 1] + temp[n//2]) / 2
    else:
        return temp[n//2]


def mode(data):
    frequency = {}
    for x in data:
        frequency[x] = frequency.get(x, 0) + 1
    max_frequency = max(frequency.values())
    for x in frequency:
        if frequency[x] == max_frequency:
            return x


def data_range(data):
    return max(data) - min(data)


def variance(data):
    m = mean(data)
    total = 0
    for x in data:
        total += (x - m) ** 2
    return total / len(data)


def standard_deviation(data):
    return math.sqrt(variance(data))

def calculate_measures(data):
    measures = {
        "Mean": mean(data),
        "Median": median(data),
        "Mode": mode(data),
        "Range": data_range(data),
        "Variance": variance(data),
        "Standard Deviation": standard_deviation(data)
    }
    return measures

def goodness(population_measure, sample_measure):
    error = 0
    error += abs(population_measure["Mean"] -
                 sample_measure["Mean"])
    error += abs(population_measure["Median"] -
                 sample_measure["Median"])
    error += abs(population_measure["Mode"] -
                 sample_measure["Mode"])
    error += abs(population_measure["Range"] -
                 sample_measure["Range"])
    error += abs(population_measure["Variance"] -
                 sample_measure["Variance"])
    error += abs(population_measure["Standard Deviation"] -
                 sample_measure["Standard Deviation"])
    return error


N = 1000          # Population size
k = 5             # Number of samples
sample_size = 50  # Size of each sample

population = []
for i in range(N):
    population.append(random.randint(1, 100))
population_measure = calculate_measures(population)

print("POPULATION MEASURES")
print("-------------------")

for key in population_measure:
    print(key, "=", round(population_measure[key], 2))

samples = []
scores = []

for i in range(k):
    sample = random.sample(population, sample_size)
    sample_measure = calculate_measures(sample)
    score = goodness(population_measure, sample_measure)
    samples.append(sample)
    scores.append(score)
    print("\nSample", i + 1)
    print("Mean =", round(sample_measure["Mean"], 2))
    print("Median =", round(sample_measure["Median"], 2))
    print("Mode =", sample_measure["Mode"])
    print("Range =", sample_measure["Range"])
    print("Variance =", round(sample_measure["Variance"], 2))
    print("Standard Deviation =",
          round(sample_measure["Standard Deviation"], 2))
    print("Goodness Error =", round(score, 2))

best_index = 0

for i in range(1, k):
    if scores[i] < scores[best_index]:
        best_index = i

print("\n--------------------------------")
print("BEST SAMPLE")
print("--------------------------------")
print("Sample Number =", best_index + 1)
print("Goodness Error =", round(scores[best_index], 2))
print("Sample =", samples[best_index])