from sklearn import tree

# 1 = smooth and 0 = bumpy 
features = [[140, 1], [130, 1], [150, 0], [140, 0]]
# 0 = apple and 1 = orange
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[150, 0]]))