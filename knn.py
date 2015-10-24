#knn.py
#k is the number of neighbors that are checked to determine which group a point belongs to
import pandas as pd
import pylab as pl
from sklearn.neighbors import KNeighborsClassifier
import numpy
from sklearn import datasets

#load the iris data set
iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
df.columns = [datasets.load_iris().feature_names]
df['type'] = iris.target
col = [colors.cnames.keys()[i] for i in df['type']] #add color for each type

plt.figure()
plt.scatter(df[df.columns[0]],df[df.columns[1]],c = col)
plt.show()


model = KNeighborsClassifier()
model.fit(iris.data, iris.target) #data, stuff we want to classify
expected = iris.target
predicted = model.predict(iris.data)
accuracy = round(float(sum(predicted == expected))/len(predicted)*100,2)
print("Model was found to be " + str(accuracy) + "% accurate")


#help with KNN can be found at http://blog.yhathq.com/posts/classification-using-knn-and-python.html