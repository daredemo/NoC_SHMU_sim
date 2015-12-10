# Copyright (C) 2015 Rene Pihlak

import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.animation as animation

def updatefig(*args):
    global index, new_data, in_data, classifierSVM, classifierKNN, axsvm, axknn
    arr = []
    if(index < len(in_data) - 5):
        new_data.append(in_data[index])
        predictedSVM = classifierSVM.predict(new_data)
        predictedKNN = classifierKNN.predict(new_data)
        for i in range(0, len(new_data)):
            arr.append(new_data[i])
        linesvm.set_ydata(arr)
        lineknn.set_ydata(arr)
        linesvm.set_xdata(range(index, index + 5))
        lineknn.set_xdata(range(index, index + 5))
        axsvm.set_ylim(-0.2, 1.3)
        axknn.set_ylim(-0.2, 1.3)
        axsvm.set_xlim(index - 0.5, index + len(arr) - 0.5)
        axknn.set_xlim(index - 0.5, index + len(arr) - 0.5)
        #print(learn_labels_names[predictedSVM[0]])
        axsvm.set_title("SVM: %s" % learn_labels_names[predictedSVM[0]])
        axknn.set_title("KNN: %s" % learn_labels_names[predictedKNN[0]])
        axsvm.figure.canvas.draw()
        axknn.figure.canvas.draw()
    index += 1
    return linesvm, lineknn,

figure = plt.figure(figsize=(27, 9))
#fig, ax = plt.subplots()
#line, = ax.step([], [], lw=2)
#ax.grid()
#xdata, ydata = [], []

axsvm = plt.subplot(2, 4, 6)
linesvm, = axsvm.step([], [], lw=2)
axsvm.grid()
xdatasvm, ydatasvm = [], []

axknn = plt.subplot(2, 4, 7)
lineknn, = axknn.step([], [], lw=2)
axknn.grid()
xdataknn, ydataknn = [], []


#MUST HAVE EQUAL NUMBER OF ALL TYPES,
#i.e., if n classified groups, then learning set size = n x m
learn_data = [
         [1,1,1,1,1],
         [1,1,1,0,0],
         [1,0,0,0,0],
         [0,0,0,0,0],
         [1,1,1,1,1],
         [0,1,1,1,0],
         [0,1,0,0,0],
         [0,0,0,0,0],
         [1,1,1,1,1],
         [0,0,1,1,1],
         [0,0,1,0,0],
         [0,0,0,0,0],
         [1,1,1,1,1],
         [1,0,0,1,1],
         [0,0,0,1,0],
         [0,0,0,0,0],
         [1,1,1,1,1],
         [1,1,0,0,1],
         [0,0,0,0,1],
         [0,0,0,0,0]]
learn_labels = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
learn_labels_names = ["dead", "dying", "ok", "perfect", "dead", "dying", "ok", "perfect", "dead", "dying", "ok", "perfect", "dead", "dying", "ok", "perfect", "dead", "dying", "ok", "perfect"]

print("LEARNING DATA:")
for index in range(0, len(learn_data)):
    print("%s: %d :: %s" % (learn_data[index], learn_labels[index], learn_labels_names[index]))
    if(index < 4):
        ax1 = plt.subplot(2, 4, index + 1)
        line1, = ax1.step([], [], lw=2)
        ax1.grid()
        line1.set_ydata(learn_data[index])
        line1.set_xdata(range(0, 5))
        ax1.set_ylim(-0.2, 1.3)
        ax1.set_xlim(-0.5, 4.5)
        ax1.set_title("Training: %s" % learn_labels_names[index])
        ax1.figure.canvas.draw()

#plt.step(range(0,5),learn_data[10], where='pre')
#plt.xlim(-0.5, len(learn_data[10]))
#plt.ylim(-0.2, 1.3)
#plt.title("Learned: %s" % (learn_labels_names[10]))
#plt.draw()

in_data = [1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,0,1,1,0,1,1,0,1,1,0,0,1,1,1]

new_data = collections.deque(maxlen=5)
new_data.append(0)
new_data.append(0)
new_data.append(0)
new_data.append(0)
new_data.append(0)
index = 0

#ax.set_ylim(-1.1, 1.1)
#ax.set_xlim(0, 10)
#del xdata[:]
#del ydata[:]
#line.set_data(xdata, ydata)


# Create a classifier: a support vector classifier
classifierSVM = svm.SVC(gamma=0.001)
classifierKNN = KNeighborsClassifier(4)

# We learn the digits on the first half of the digits
classifierSVM.fit(learn_data, learn_labels)
classifierKNN.fit(learn_data, learn_labels)

#predicted = classifier.predict(data)
movis = animation.writers['avconv']
movi = movis(fps=1, bitrate=1800)
ani = animation.FuncAnimation(figure, updatefig, interval=500, blit=True)
ani.save('mlmultipletest.mp4', writer=movi, fps=1, bitrate=1800)
#print("ANALYSED DATA:")
#for index in range(0, len(data)):
#    print("%s: %s" % (data[index], learn_labels_names[predicted[index]]))

plt.show()
