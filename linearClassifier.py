import scipy.io
import numpy as np
import psutil, os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

mat = scipy.io.loadmat('dataset.mat')
train_data = mat.get("Train_Data")
train_data = np.array(train_data)
CLASS_COUNT = 5
x_vectors = []
class_number = []
x_class1 = []
x_class2 = []
x_class3 = []
x_class4 = []
x_class5 = []


# print len(train_data)
def memory_usage_psutil():  # return KB
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem * 1048576 / 1024


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
from sklearn import svm
import time

start_time = int(round(time.time() * 1000))
lin_clf = svm.SVC(decision_function_shape='ovr')
lin_clf.fit(x_vectors, class_number)
print lin_clf
test_data = mat.get("Test_Data")

index = 0
correct_count = 0.0
confusion_matrix = np.array(np.zeros(shape=(CLASS_COUNT, CLASS_COUNT)))
output_file = open("linear-classifier.txt", "w")
for data in test_data:
    index += 1
    print index
    my_x = data[0:51]
    class_rec = data[51:]
    which_class = -1
    for i in range(0, len(class_rec)):
        if class_rec[i] == 1:
            which_class = i
            break

    my_x = np.array(my_x).reshape(1, -1)
    # my_x = np.array(np.matrix(my_x).T)
    dec = lin_clf.decision_function(my_x)
    print dec
    # class_choose = dec.shape[1]
    class_choose = np.argmax(dec[0])
    confusion_matrix[which_class][class_choose] += 1
    if which_class == class_choose:
        correct_count += 1

print "correct count: ", correct_count

print "Accuracy: ", (correct_count / len(test_data)) * 100
print "confusion matrix: "
print confusion_matrix
end_time = int(round(time.time() * 1000))
print  "classify time: ", end_time - start_time
print "memory usage: ", memory_usage_psutil(), " KB"

output_file.write("correct count: " + str(correct_count))
output_file.write("\nAccuracy: " + str((correct_count / len(test_data)) * 100))
output_file.write("\nconfusion matrix: \n")
output_file.write(str(confusion_matrix))
output_file.write("\nclassify time: " + str(end_time - start_time))
output_file.write("\nmemory usage: " + str(memory_usage_psutil()) + " KB")
output_file.close()
