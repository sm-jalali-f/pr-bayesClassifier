import scipy.io
import numpy as np
import math
import time,os,psutil
from sklearn.neighbors import KNeighborsClassifier

K_NEIGHBOUR = [1,3,5,7]
CLASS_COUNT = 5
def memory_usage_psutil():#return KB
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem * 1048576 /1024

def calculate_euclidean_distance(x1, x2):
    result = 0.0;
    for i in range(0, len(x1)):
        result += math.pow(x1[i] - x2[i], 2)
    return math.sqrt(result)


mat = scipy.io.loadmat('dataset.mat')

train_data = mat.get("Train_Data")
train_data = np.array(train_data)
x_vectors = []
class_number = []

for i in range(0, len(train_data), 1):
    x = []
    class_id = -1
    for j in range(0, len(train_data[i]), 1):
        if j > len(train_data[i]) - 6:
            # print train_data[i][j]
            if train_data[i][j] == 1:
                class_number.append(j - len(train_data[i]) + 6)
                class_id = j - len(train_data[i]) + 6
                continue
        else:
            x.append(train_data[i][j])
    x_vectors.append(x)

count_of_each = [0, 0, 0, 0, 0]
for i in range(0, len(class_number)):
    if class_number[i] == 1:
        count_of_each[0] += 1
    elif class_number[i] == 2:
        count_of_each[1] += 1
    elif class_number[i] == 3:
        count_of_each[2] += 1
    elif class_number[i] == 4:
        count_of_each[3] += 1
    elif class_number[i] == 5:
        count_of_each[4] += 1

test_data = mat.get("Test_Data")
