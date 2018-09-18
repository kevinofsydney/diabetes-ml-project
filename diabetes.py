import csv
import numpy as np

filename = "diabetes.csv"
DEBUG = False

# There are 9 columns in this CSV. The first 8 are features, titled as so:
# - Pregnancies
# - Glucose
# - BloodPressure
# - SkinThickness
# - Insulin
# - BMI
# - DiabetesPedigreeFunction
# - Age

# The last column is whether they have diabetes (1) or not (0).


# First, we read in the data
def csv_loader(filename):
    pid_data = []

    with open(filename, 'r', newline='') as csvfile:
        raw_data = csv.reader(csvfile, delimiter=',')

        for index, row in enumerate(raw_data):
            # Skip the row of labels
            if index == 0:
                continue

            # Converting the string data into integers
            # For readability, assign 1s and 0s directly instead of converting to float
            for index2, item in enumerate(row):
                if item == "1":
                    row[index2] = 1
                elif item == "0":
                    row[index2] = 0
                else:
                    row[index2] = float(item)

            pid_data.append(row)

    return pid_data


# This function stratifies and divides the data into 10 folds, making sure there are 'yes' and 'no'
#   data points in each fold.
def stratifier(pid_data):
    class_yes = []
    class_no = []

    # Separate the yes and no data points
    while len(pid_data) > 0:
        current = pid_data.pop()
        if current[8] == 1:
            class_yes.append(current)
        else:
            class_no.append(current)

    folds = [[], [], [], [], [], [], [], [], [], []]
    cursor = 0

    # Add one row both classes into each fold until exhausted
    # TODO: Note that this can also be performed with numpy.split
    while len(class_no) > 0:
        current = class_no.pop()
        folds[cursor].append(current)

        if len(class_yes) > 0:
            current = class_yes.pop()
            folds[cursor].append(current)

        if cursor == 9:
            cursor = 0
        else:
            cursor += 1

    # This code block exports the folds as a CSV, just for checking purposes
    if DEBUG:
        with open("pid-folds.csv", 'w') as outfile:
            writer = csv.writer(outfile, delimiter=',')

            index = 1
            for f in folds:
                writer.writerow(["fold" + str(index)])
                for pid in f:
                    writer.writerow(pid)

                writer.writerow("")
                index += 1

    return folds


def label_separator(combined_data):
    data = []
    labels = []

    for i in combined_data:
        data.append(i[:8])
        labels.append(i[8])

    return data, labels


def knn(trainbatch, testbatch, k):
    traindata, trainlabel = label_separator(trainbatch)
    testdata, testlabel = label_separator(testbatch)

    result = []

    for testrow in testdata:
        label_tally = [0, 0] # label_tally[0] = No, label_tally[1] = Yes
        distance = []

        # Calculate the Euclidean distance between the test row and all of the training rows
        for trainrow in traindata:
            distance.append(np.linalg.norm(np.array(testrow) - np.array(trainrow)))

        # np.argsort returns the array that sorts the distance array
        nearest = np.argsort(distance)

        # Retrieve the labels for the k-nearest neighbours
        for j in range(k):
            label_tally[trainlabel[nearest[j]]] += 1

        # Count the number of neighbors to label the test datapoint
        if label_tally[0] > label_tally[1]:  # Most neighbors are 'no' datapoints
            result.append(0)
        else:
            result.append(1)

    # XOR the results with the test labels and count the number of Falses to determine accuracy.
    tmp = np.logical_xor(np.array(result), np.array(testlabel))
    count = 0
    for result in tmp:
        if result == False:
            count += 1

    if DEBUG:
        print("%d correctly labelled out of %d (%.2f%%)" % (count, len(testlabel), count/len(testlabel)*100))

    # Returns the accuracy of this fold in %
    return count/len(testlabel)


if __name__ == "__main__":
    filename = "diabetes.csv"
    pid_data = csv_loader(filename)  # PID stands for Pima Indian Diabetes data
    folds = stratifier(pid_data)

    k = 5
    knn_accuracy = []

    print("Running k-Nearest Neighbour (k = %d) with 10-fold cross-validation..." % k)

    # Using one fold as the test set and the other 9 together as the training set, run k-nn
    for i in range(10):

        if DEBUG:
            print("Testing fold", i, "of 10...")

        pid_copy = folds[:]
        testbatch = pid_copy[i]
        pid_copy.remove(testbatch)
        trainbatch = [row for fold in pid_copy for row in fold] # Convert 3D list into 2D list
        result = knn(trainbatch, testbatch, k)
        knn_accuracy.append(result)

    print("Average accuracy of %.2f%% across all ten folds." % (np.average(knn_accuracy)*100))
