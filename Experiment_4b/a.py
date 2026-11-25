#Write a python script to measure the proximity measure for the given dataset comprising mixed type attributes
# Inputs: Ordinal, Nominal, Asymmetric binary, numerical (dataset attributes)

data=[
    #ordinal    #Nominal      #Asymmetric binary        #Numerical
    ["Low"   ,  "Red"   ,        1,                         20],
    ["Medium",  "Blue"  ,        0,                         40],
    ["High"  ,  "Yellow",        0,                         60],
    ["Medium",  "Green" ,        1,                         30]
]

ordinal_rank={
    "Low":1,
    "Medium":2,
    "High":3
}

#Nominal dissimilarity
def nominal_distance(x,y):
    if x==y:
        return 0;
    else:
        return 1

#Numerical dissimilarity
def numerical_distance(x,y,maximum,minimum):
    if maximum==minimum:
        return 0
    else:
        return abs(x-y)/maximum-minimum

#ordinal dissimilarity
def ordinal_distance(x,y):
    rx=ordinal_rank[x]
    ry=ordinal_rank[y]

    max_rank=max(ordinal_rank.values())
    min_rank=min(ordinal_rank.values())

    zx=(rx-min_rank)/(max_rank-min_rank)
    zy=(ry-min_rank)/(max_rank-min_rank)

    return abs(zx-zy)

#Asymmetric Binary
def asymmetric_binary_distance(x,y):
    if x==0 and y==0:
        delta=0
        d=0

    if x==1 and y==0:
        delta=1
        d=0

    else:
        delta=1
        d=1

    return delta,d

#Numerical attribute range
numerical_values=[]
for row in data:
    numerical_values.append(row[3])
minimum=min(numerical_values)
maximum=max(numerical_values)

#Mixed dissimilarity
def mixed_distance(row1,row2):
    numerator=0
    denominator=0

    d=ordinal_distance(row1[0],row2[0])
    delta=1
    numerator=numerator+delta*d
    denominator=denominator+delta

    d=nominal_distance(row1[1],row[1])
    delta=1
    numerator=numerator+delta*d
    denominator=denominator+delta

    delta,d=asymmetric_binary_distance(row1[2],row2[2])
    numerator=numerator+delta*d
    denominator=denominator=delta

    d=numerical_distance(row1[3],row2[3],maximum,minimum)
    delta=1
    numerator=numerator+delta*d
    denominator=denominator+delta

    if denominator==0:
        return 0
    
    return numerator/denominator




n=len(data)
matrix = []
for i in range(n):
    temp=[]
    for j in range(n):
        d=mixed_distance(data[i],data[j])
        temp.append(round(d,3))
    matrix.append(temp)

print("Dataset:\n")
print("Object\tOrdinal\t\tNominal\t\tAsym.Binary\tNumerical")
for i in range(n):
    print(
        "P"+str(i+1),
        "\t",
        data[i][0],
        "\t\t",
        data[i][1],
        "\t\t",
        data[i][2],
        "\t\t",
        data[i][3]
    )

print("\nFinal Dissimilarity Matrix:\n")
print("\t",end="")
for i in range(n):
    print("P"+str(i+1), end="\t")
print()


for i in range(n):
    print("P"+str(i+1), end="\t")
    for j in range(n):
        print(matrix[i][j],end="\t")
    print()