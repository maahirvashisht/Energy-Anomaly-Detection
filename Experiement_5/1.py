# Write a python script to determine the size of a sample:
#     a) Random Sampling
#     b) Cluster based sampling
#     c) Stride based sampling
# (write your own functions with required parameters)


def random_sampling(population_size,sample_percentage):
    sample_size=int((sample_percentage/100)*population_size)
    return sample_size

def cluster_sampling(total_clusters,selected_clusters,elements_per_clusters):
    if selected_clusters>total_clusters:
        return "Error"
    sample_size=selected_clusters*elements_per_clusters
    return sample_size

def stride_sampling(population_size,stride):
    if stride<=0:
        return "stride must be greater than 0"
    sample_size=(population_size+stride-1)//stride
    return sample_size



population_size=int(input("Enter population size: "))
percentage=float(input("Enter sample percentage for random sampling: "))

random_size=random_sampling(
    population_size,
    percentage
)
print("Random sampling size: ",random_size)



total_clusters=int(input("\nEnter total numbers of clusters: "))
selected_clusters=int(input("Enter number of clusters to select: "))
elements_per_cluster=int(input("Enter number of elements in each cluster: "))
cluster_size=cluster_sampling(
    total_clusters,
    selected_clusters,
    elements_per_cluster
)
print("Cluster Based Sampling Size =", cluster_size)



stride = int(input("\nEnter stride value: "))
stride_size = stride_sampling(
    population_size,
    stride
)
print("Stride Based Sampling Size =", stride_size)