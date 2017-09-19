import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# print("Iris feature_names : \n", iris.feature_names, "\n")
# print("Iris target names : \n", iris.target_names, "\n")
# print("Iris Data : \n" , iris.data[0])
# print("Iris Target : \n" , iris.target[0])

#for removing one of each kind for testing
test_idx = [0,50,100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print("Test Data : ", test_data)
print("Test Target : ", test_target)
print("Prediction : ", clf.predict(test_data))