import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# #read the csv dataset, that sep thing is used to mention how the data are septd
# df = pd.read_csv("student-mat.csv",sep = ";")
# df = df[["G1","G2","G3","studytime","failures","absences"]]
# #drop drops column if 1 and row if 0
# X = df.drop(["G3"],1)
# Y = df["G3"]
# #numpy array are damn amazing
# X = np.array(X)
# Y = np.array(Y)
# #train test split returns a training and test collection of data
# x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)
# """
# best =0
# for i in range(30):
#     x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train,y_train)
#     acc = linear.score(x_test,y_test)
#     print(acc)
#     if acc>best:
#         best = acc
#         #pickle is used to dump an ML model into a pickle file
#         with open("student_model.pickle","wb") as f:
#             pickle.dump(linear,f)"""
#
# #This is how we load it
# pickle_ld = open("student_model.pickle","rb")
# linear = pickle.load(pickle_ld)
#
# print("Predictions...")
# #Actual heart of prediction
# predictions = linear.predict(x_test)
#
# for i in range(len(predictions)):
#     print(int(predictions[i]),y_test[i],sep="-------------")
#
#
# #some matplotlib stuffs. and it is cool
# style.use("ggplot")
# domain = "absences"
# codomain = "G3"
# pyplot.scatter(df[domain],df[codomain])
# pyplot.xlabel(domain)
# pyplot.ylabel(codomain)
# pyplot.show()

#############################################
# KNN
#############################################

df = pd.read_csv("car.data")
print(df.head())
# returns an label encoder object
LE = preprocessing.LabelEncoder()

#using that object to encode the values of cols
#the fit_transform thing returns a np array
buying = LE.fit_transform(list(df["buying"]))
maint = LE.fit_transform(list(df["maint"]))
door = LE.fit_transform(list(df["door"]))
persons = LE.fit_transform(list(df["persons"]))
lug_boot = LE.fit_transform(list(df["lug_boot"]))
safety = LE.fit_transform(list(df["safety"]))
cls = LE.fit_transform(list(df["class"]))

#okay, so u can use xy as list or numpy array,but both data type should be same to use that train test split thing
X = list(zip(buying,maint,door,persons,lug_boot,safety))
Y = list(cls)
x_train, x_test, y_train,_y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

"""best =0
for i in range(30):
    x_train, x_test, y_train,_y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)
    knn_model = KNeighborsClassifier(n_neighbors=9)
    knn_model.fit(x_train,y_train)
    acc = knn_model.score(x_test,_y_test)
    print(acc)
    if acc > best:
        best = acc
        with open("carknnmodel.pickle","wb") as f:
            pickle.dump(knn_model,f)"""

pickle_in = open("carknnmodel.pickle","rb")
knn_model = pickle.load(pickle_in)
acc = knn_model.score(x_test,_y_test)
print(acc)
