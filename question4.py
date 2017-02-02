import scipy.io
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import math
import os
import psutil

CLASS_COUNT = 5
BASE_LANDA = 0.01
def memory_usage_psutil():#return KB
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem * 1048576 /1024

def calculate_sigma_p(all_instance_count, class_count, all_sigma, each_class_count):
    sum_sigma = np.zeros(shape=(len(all_sigma[0]), len(all_sigma[0])))
    for i in range(0, class_count):
        sum_sigma = sum_sigma + np.matrix((each_class_count[i] - 1) * np.array(all_sigma[i]))
    return np.matrix(np.array(sum_sigma) / (all_instance_count - class_count))


def friedmanSigma(sigma, landa, class_instance_count, all_instance_count, class_count, sigma_p):
    x1 = np.matrix((1 - landa) * (class_instance_count - 1) * np.array(sigma))
    x2 = np.matrix(landa * (all_instance_count - class_count) * np.array(sigma_p))
    x3 = (1 - landa) * class_instance_count + landa * all_instance_count

    return np.matrix(np.array(x1 + x2) / x3)


def calculate_pdf(x_data, mean, sigma, pi_class):
    determinant = det(sigma)
    if determinant == 0:
        m = 10 ^ -6
        sigma += np.eye(sigma.shape[0]) * m
        determinant = det(sigma)
    if determinant < 0:
        return -1000
    temp1 = -np.log(determinant)/2
    temp2 = np.array(np.dot(np.dot((x_data - mean).T, inv(sigma)), x_data - mean))
    temp2 = -temp2[0][0] /2
    temp3 =  np.log(pi_class)
    return temp1 + temp2 + temp3


def calculate_mean(class_data, instance_count):
    mean = [sum(x) for x in zip(*class_data)]
    mean = [x / instance_count for x in mean]
    mean = np.transpose(np.matrix(mean))
    return mean


def calculate_sigma(class_data, class_mean, instance_count):
    sigma = np.zeros(shape=(len(class_data[0]), len(class_data[0])))
    for x in class_data:
        sigma = sigma + np.dot((np.transpose(np.matrix(x)) - class_mean),
                               np.transpose((np.transpose(np.matrix(x)) - class_mean)))
    sigma = np.matrix(np.array(sigma) / instance_count)
    return sigma


import time
start_time = int(round(time.time() * 1000))


mat = scipy.io.loadmat('dataset.mat')

train_data = mat.get("Train_Data")
train_data = np.array(train_data)
x_vectors = []
class_number = []
x_class1 = []
x_class2 = []
x_class3 = []
x_class4 = []
x_class5 = []
# print len(train_data)
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
    if class_id == 1:
        x_class1.append(x)
    elif class_id == 2:
        x_class2.append(x)
    elif class_id == 3:
        x_class3.append(x)
    elif class_id == 4:
        x_class4.append(x)
    elif class_id == 5:
        x_class5.append(x)

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

mean1 = calculate_mean(class_data=x_class1, instance_count=count_of_each[0])
sigma1 = calculate_sigma(class_data=x_class1, class_mean=mean1, instance_count=count_of_each[0])
mean2 = calculate_mean(class_data=x_class2, instance_count=count_of_each[1])
sigma2 = calculate_sigma(class_data=x_class2, class_mean=mean2, instance_count=count_of_each[1])
mean3 = calculate_mean(class_data=x_class3, instance_count=count_of_each[2])
sigma3 = calculate_sigma(class_data=x_class3, class_mean=mean3, instance_count=count_of_each[2])
mean4 = calculate_mean(class_data=x_class4, instance_count=count_of_each[3])
sigma4 = calculate_sigma(class_data=x_class4, class_mean=mean4, instance_count=count_of_each[3])
mean5 = calculate_mean(class_data=x_class5, instance_count=count_of_each[4])
sigma5 = calculate_sigma(class_data=x_class5, class_mean=mean5, instance_count=count_of_each[4])

test_data = mat.get("Test_Data")

landa_r = 0.8
landa_s = 1.0
LANDA = 0.68
test_each_class_count = []
for i in range(0, CLASS_COUNT):
    test_each_class_count.append(0)
sigma_p = calculate_sigma_p(all_instance_count=len(train_data), class_count=CLASS_COUNT,
                            all_sigma=[sigma1, sigma2, sigma3, sigma4, sigma5], each_class_count=count_of_each)
fried_sigma1 = friedmanSigma(sigma=sigma1, landa=LANDA, class_count=CLASS_COUNT,
                             class_instance_count=count_of_each[0],
                             all_instance_count=len(train_data), sigma_p=sigma_p)
fried_sigma2 = friedmanSigma(sigma=sigma2, landa=LANDA, class_count=CLASS_COUNT,
                             class_instance_count=count_of_each[1],
                             all_instance_count=len(train_data), sigma_p=sigma_p)
fried_sigma3 = friedmanSigma(sigma=sigma3, landa=LANDA, class_count=CLASS_COUNT,
                             class_instance_count=count_of_each[2],
                             all_instance_count=len(train_data), sigma_p=sigma_p)
fried_sigma4 = friedmanSigma(sigma=sigma4, landa=LANDA, class_count=CLASS_COUNT,
                             class_instance_count=count_of_each[3],
                             all_instance_count=len(train_data), sigma_p=sigma_p)
fried_sigma5 = friedmanSigma(sigma=sigma5, landa=LANDA, class_count=CLASS_COUNT,
                             class_instance_count=count_of_each[4],
                             all_instance_count=len(train_data), sigma_p=sigma_p)
correct_count = 0.0
# confusion_matrix = np.array(np.zeros(shape=(CLASS_COUNT, CLASS_COUNT+1)))
confusion_matrix = []
output_file = open("question4-landar8.txt","w")
for k in range(0, CLASS_COUNT):
    temp = []
    for j in range(0, CLASS_COUNT + 1):
        temp.append(0.0)
    confusion_matrix.append(temp)

for data in test_data:
    my_x = data[0:51]
    class_rec = data[51:]
    which_class = -1
    for i in range(0, len(class_rec)):
        if class_rec[i] == 1:
            which_class = i
            test_each_class_count[i] += 1
            break
    my_x = (np.matrix(my_x)).T.conj()
    pdf = [calculate_pdf(my_x, mean=mean1, sigma=fried_sigma1, pi_class=0.2),
           calculate_pdf(my_x, mean=mean2, sigma=fried_sigma2, pi_class=0.2),
           calculate_pdf(my_x, mean=mean3, sigma=fried_sigma3, pi_class=0.2),
           calculate_pdf(my_x, mean=mean4, sigma=fried_sigma4, pi_class=0.2),
           calculate_pdf(my_x, mean=mean5, sigma=fried_sigma5, pi_class=0.2)]

    pdf.append(sum(pdf) * ((landa_s - landa_r) / landa_s))

    # print pdf
    class_choos = np.argmax(pdf)
    confusion_matrix[which_class][class_choos] += 1
    if which_class == class_choos:
        correct_count += 1

print "================= landa=", LANDA, " ================="

print "correct classify count = ", correct_count

accuracy = correct_count / len(test_data)
print "accuracy = ", accuracy * 100, "%"
print "confusion matrix : "

print confusion_matrix
output_file.write(str(confusion_matrix))
each_class_precision = []
each_class_recall = []
for i in range(0, CLASS_COUNT):
    each_class_precision.append(confusion_matrix[i][i] / test_each_class_count[i])
    each_class_recall.append(confusion_matrix[i][i] / len(test_data))
print "precision: ",each_class_precision


print "recal: ",each_class_recall

print "memory usage: " , memory_usage_psutil() ," KB"

end_time = int(round(time.time() * 1000))

print  "classify time: " ,end_time - start_time
output_file.write("\ncorrect classify count = "+str(correct_count))
output_file.write("\naccuracy = "+str(accuracy * 100)+ "%")
output_file.write("\nconfusion matrix : \n")
output_file.write(str(confusion_matrix))
output_file.write("\nprecision: \n")
output_file.write(str(each_class_precision))
output_file.write("\nrecal: \n")
output_file.write(str(each_class_recall))
output_file.write("\nmemory usage: " +str(memory_usage_psutil()) + " KB")
output_file.write( "\nclassify time: " +str(end_time - start_time))
output_file.close()