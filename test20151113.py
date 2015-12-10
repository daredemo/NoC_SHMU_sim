# Copyright (C) 2015 Rene Pihlak

import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn import svm, metrics
import matplotlib.animation as animation

def updatefig(*args):
    global index, new_data, line, in_data, classifier
    arr = []
    if(index < len(in_data) - 5):
        new_data.append(in_data[index])
        predicted = classifier.predict(new_data)
        for i in range(0, len(new_data)):
            arr.append(new_data[i])
        line.set_ydata(arr)
        line.set_xdata(range(index, index + 5))
        ax.set_ylim(-0.2, 1.3)
        ax.set_xlim(index - 0.5, index + len(arr) - 0.5)
        print(learn_labels_names[predicted[0]])
        ax.set_title(learn_labels_names[predicted[0]])
        ax.figure.canvas.draw()
    index += 1
    return line,

fig, ax = plt.subplots()
line, = ax.step([], [], lw=2)
ax.grid()
xdata, ydata = [], []


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

ax.set_ylim(-1.1, 1.1)
ax.set_xlim(0, 10)
del xdata[:]
del ydata[:]
line.set_data(xdata, ydata)


# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(learn_data, learn_labels)

#predicted = classifier.predict(data)
movis = animation.writers['avconv']
movi = movis(fps=1, bitrate=1800)
ani = animation.FuncAnimation(fig, updatefig, interval=500, blit=True)
ani.save('moviefile.mp4', writer=movi, fps=1, bitrate=1800)
#print("ANALYSED DATA:")
#for index in range(0, len(data)):
#    print("%s: %s" % (data[index], learn_labels_names[predicted[index]]))

plt.show()
