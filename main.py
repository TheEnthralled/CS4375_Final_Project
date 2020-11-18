
import sys

# Attribute are represented by (str, list, bool) where the 1st string is the name, the second the range, and the third
# boolean represents if the attribute is discrete or not
from typing import List
from sklearn import tree

if __name__ == '__main__':
    attr_file_name  = sys.argv[1]
    train_file_name = sys.argv[2]

    attr_file = open(attr_file_name)
    attr_names = []
    attr_range = [] #ex continuous, discrete
    for line in attr_file:
        l = line.split()
        attr_names.append(l[0])

    train_samples = []
    class_samples = []
    train_file = open(train_file_name)
    for line in train_file:
        l = line.split()
        class_samples.append(int(l.pop()))
        for i in range(0, len(attr_names)-1):
            l[i] = float(l[i])
        train_samples.append(l)


    clf = tree.DecisionTreeClassifier()
    clf.fit(train_samples, class_samples)

    predict = clf.predict(train_samples)
    num_correct = 0
    for i in range(0, len(predict)):
        if predict[i] == class_samples[i]:
            num_correct+=1
    print(float(num_correct)/len(class_samples)*100)



    #tree.plot_tree(clf)
    #print("done")