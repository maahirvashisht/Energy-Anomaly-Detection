# c) Numerical (Manhatten and Euclidean)

def manhattan_distance(a,b):
    distance=0
    for i in range(len(a)):
        distance=distance+abs(a[i]-b[i])
    return distance

def euclidean_distance(a,b):
    total=0
    for i in range(len(a)):
        total=total+(a[i]-b[i])**2
    return total**0.5

data = [
    [2, 4, 5],
    [5, 1, 7],
    [3, 4, 6]
]

n=len(data)

manhattan_matrix=[]

for i in range(n):
    row=[]
    for j in range(n):
        d=manhattan_distance(data[i],data[j])
        row.append(round(d,2))
    manhattan_matrix.append(row)

print("Manhattan Dissimilarity Matrix: ")

for row in manhattan_matrix:
    print(row)

euclidean_matrix=[]

for i in range(n):
    row=[]
    for j in range(n):
        d=euclidean_distance(data[i],data[j])
        row.append(round(d,2))
    euclidean_matrix.append(row)

print("Euclidean Dissimilarity Matrix: ")

for row in euclidean_matrix:
    print(row)