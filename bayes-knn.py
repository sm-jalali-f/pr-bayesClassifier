import scipy.io
import numpy as np
import math
import time, os, psutil

K_NEIGHBOUR = 1
CLASS_COUNT = 5


def memory_usage_psutil():  # return KB
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem * 1048576 / 1024


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

output_file = open("knn-classifier.txt", 'w')
k_list = [9, 10]
for neighbour_count in range(0, len(k_list)):
    start_time = int(round(time.time() * 1000))
    test_each_class_count = []
    for i in range(0, CLASS_COUNT):
        test_each_class_count.append(0)
    confusion_matrix = []
    correct_count = 0.0
    for k in range(0, CLASS_COUNT):
        temp = []
        for j in range(0, CLASS_COUNT):
            temp.append(0.0)
        confusion_matrix.append(temp)
    K_NEIGHBOUR = k_list[neighbour_count]
    index = 0
    for data in test_data:
        index += 1
        print index
        my_x = data[0:51]
        class_rec = data[51:]
        which_class = -1
        for i in range(0, len(class_rec)):
            if class_rec[i] == 1:
                which_class = i
                test_each_class_count[i] += 1
                break
        distance_list = []
        for i in range(0, len(x_vectors)):
            distance_list.append((calculate_euclidean_distance(my_x, x_vectors[i]), class_number[i]))
        k_instance_class = []
        for i in range(0, CLASS_COUNT):
            k_instance_class.append(0)

        distance_list = sorted(distance_list, key=lambda x: x[0])
        for i in range(0, K_NEIGHBOUR):
            which_min = distance_list[i]
            k_instance_class[distance_list[i][1] - 1] += 1

        class_choose = np.argmax(k_instance_class)
        confusion_matrix[which_class][class_choose] += 1
        # print "class choose = ", class_choose
        # print "which class = ", which_class
        if which_class == class_choose:
            correct_count += 1
    print "\n================================================================="
    end_time = int(round(time.time() * 1000))


    print  "classify time: ", end_time - start_time
    print "memory usage: ", memory_usage_psutil(), " KB"
    print "k-neighbour: ", K_NEIGHBOUR
    print "correct count: ", correct_count
    print "Accuracy: ", (correct_count / len(test_data)) * 100
    print "confusion: "
    print confusion_matrix
    output_file.write("===================================================")
    output_file.write("\nk-neighbour: ")
    output_file.write(str(K_NEIGHBOUR))
    output_file.write("\ncorrect count: ")
    output_file.write(str(correct_count))
    output_file.write("\nAccuracy: ")
    output_file.write(str((correct_count / len(test_data)) * 100))
    output_file.write("\nconfusion: ")
    output_file.write(str(confusion_matrix))
    output_file.write("\nclassify time: ")
    output_file.write(str(end_time - start_time))
    output_file.write("\nMemory usage:")
    output_file.write(str(memory_usage_psutil()) + " KB")
output_file.close()
