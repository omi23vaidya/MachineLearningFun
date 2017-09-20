from scipy.spatial import distance
def euc(a,b):
	return distance.euclidean(a,b)

class ScrappyKNN():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		predictions = []
		for row in X_test:
				label = self.closest(row)
				predictions.append(label)
		return predictions

	def closest(self, row):
		best_distance = euc(row, self.X_train[0])
		best_index = 0
		for i in range(1, len(self.X_train)):
			dist = euc(row, self.X_train[i])
			if dist < best_distance:
				best_distance = dist
				best_index = i
		return self.y_train[best_index]

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

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()
my_classifier = ScrappyKNN()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print("Predictions : " ,predictions)

from sklearn.metrics import accuracy_score
print("Accuracy : ", accuracy_score(y_test, predictions))