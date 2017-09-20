##Writing a pipeline

#import a dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

print("X is : ", X)
print("Y is : \n", y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

print("X_train is : ", X_train)
print("X_test is : \n", X_test)

print("y_train is : ", y_train)
print("y_test is : \n", y_test)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print("Predictions : " ,predictions)

from sklearn.metrics import accuracy_score
print("Accuracy : ", accuracy_score(y_test, predictions))