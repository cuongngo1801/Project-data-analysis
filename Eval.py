import math
import numpy as np

matrix = np.random.randint(0,100,size=(6,6))

##compute SSE for each cluster
def SSE(feature_matrix):
    mean_vector = feature_matrix.mean(axis = 0)
    squared_mean_vector = np.square(mean_vector)
    squared_mean = np.sum(squared_mean_vector)
    SSE = 0.0
    for i in range(np.shape(feature_matrix)[0]):
        value1_vector = np.square(feature_matrix[i][:])
        value1 = np.sum(value1_vector)
        SSE = SSE + (value1 - squared_mean)
    return SSE
#def create_eval_matrix(cluster, bin ):

def find_entropy(matrix):
    total_entropy = 0
    sums = []
    entropy = []
    total_sum = np.sum(matrix) 
    for i in range(6):
        minus_entropy = 0
        sums.append(np.sum(matrix[i][:]))
        for j in range(6):
            a = matrix[i][j]/sums[i]
            minus_entropy = minus_entropy + a * math.log(a)
            local_entropy = - minus_entropy
        entropy.append(local_entropy)
        total_entropy = total_entropy + (sums[i]/total_sum) * entropy[i]
    return total_entropy

def find_purity(matrix):
    n = np.sum(matrix)
    maxs = 0 
    for i in range(6):
        maxs = maxs + np.max(matrix[i][:])
    purity = (1/n) * maxs
    return purity

print(find_purity(matrix))       
