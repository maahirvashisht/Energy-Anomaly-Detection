# a) Nominal

def nominal_dissimilarity(a,b):
    mismatch=0
    for i in range(len(a)):
        if a[i]!=b[i]:
            mismatch=mismatch+1

    return mismatch/len(a)\


data=[
    ["Red","Small","Circle"],
    ["Blue","Small","Circle"],
    ["Red","Large","Square"]
]

n=len(data)

matrix=[]

for i in range(n):
    row=[]
    for j in range(n):
        d=nominal_dissimilarity(data[i],data[j])
        row.append(round(d,2))

    matrix.append(row)

print("Nominal Dissimilarity Matrix: ")
for row in matrix:
    print(row)