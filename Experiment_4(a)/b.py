# i) Binary (Symmetric)

# def symmetric_binary_dissimilarity(a,b):
#     mismatch=0
#     for i in range(len(a)):
#         if a[i]!=b[i]:
#             mismatch=mismatch+1
#     return mismatch/len(a)

# data=[
#     [1,0,1,1],
#     [1,1,0,1],
#     [0,0,1,1]
# ]

# n=len(data)

# matrix=[]

# for i in range(n):
#     row=[]

#     for j in range(n):
#         d=symmetric_binary_dissimilarity(data[i],data[j])
#         row.append(round(d,2))

#     matrix.append(row)

# print("Symmetric Binary dissimilarity: ")

# for row in matrix:
#     print(row)





# ii) Binary (Assymetric)

def asymmetric_binary_dissimilarity(a,b):
    m11=0
    m10=0
    m01=0

    for i in range(len(a)):
        if a[i]==1 and b[i]==1:
            m11=m11+1
        if a[i]==1 and b[i]==0:
            m10=m10+1
        if a[i]==0 and b[i]==1:
            m01=m01=1

    total = m11+m10+m01

    if total==0:
        return 0

    return(m10+m01)/total

data=[
    [1,0,1,1],
    [1,1,0,1],
    [0,0,1,1]
]

n=len(data)

matrix=[]

for i in range(n):
    row=[]

    for j in range(n):
        d=asymmetric_binary_dissimilarity(data[i],data[j])
        row.append(round(d,2))

    matrix.append(row)

print("Symmetric Binary dissimilarity: ")

for row in matrix:
    print(row)