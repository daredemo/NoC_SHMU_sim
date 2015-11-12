import numpy as np
from sklearn import svm, metrics

learn_data = [
         [1,1,1,1,1],
         [1,1,1,0,0],
         [1,0,0,0,0],
         [0,0,0,0,0]]
learn_labels = [0,1,2,3]
learn_labels_names = ["dead", "dying", "ok","perfect"]

print("LEARNING DATA:")
for index in range(0, len(learn_data)):
    print("%s: %d :: %s" % (learn_data[index], learn_labels[index], learn_labels_names[index]))

data = [
         [1,1,1,1,1],
         [1,1,1,0,0],
         [1,0,0,0,0],
         [0,0,0,0,0],
         [1,1,1,1,1],
         [1,1,1,0,0],
         [1,0,0,1,0],
         [0,0,0,0,1],
         [1,0,1,1,0],
         [1,1,0,1,1],
         [0,0,1,1,1]]

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(learn_data, learn_labels)

predicted = classifier.predict(data)

print("ANALYSED DATA:")
for index in range(0, len(data)):
    print("%s: %s" % (data[index], learn_labels_names[predicted[index]]))
