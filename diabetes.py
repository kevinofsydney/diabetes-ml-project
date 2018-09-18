import os
import csv
import numpy

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


# This function stratifies the data into 10 folds, making sure there are 'yes' and 'no'
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

def knn(trainbatch, testbatch, k):

    return

if __name__ == "__main__":
    filename = "diabetes.csv"
    pid_data = csv_loader(filename)  # PID stands for Pima Indian Diabetes data
    folds = stratifier(pid_data)

    # TODO: run for multiple values of k
    k = 5

    # Using one fold as the test set and the other 9 together as the training set, run k-nn
    for i in range(10):
        pid_copy = folds[:]
        testbatch = pid_copy[i]
        pid_copy.remove(testbatch)
        trainbatch = [row for fold in pid_copy for row in fold] # Convert 3D list into 2D list
        knn(trainbatch, testbatch, k)
