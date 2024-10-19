# mlp for multi-label classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

# get the dataset
def get_dataset():
	X, y = make_multilabel_classification(n_samples=100, n_features=10, n_classes=3, n_labels=2, random_state=1)
	return X, y

# load dataset
X, y = get_dataset()

for i in range(10):
 print(X[i], y[i])

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

results = list()
n_inputs, n_outputs = X.shape[1], y.shape[1]

print (n_inputs)
print (n_outputs)

# define evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

fold =1
for train_ix, test_ix in cv.split(X):
  # prepare data
  X_train, X_test = X[train_ix], X[test_ix]
  y_train, y_test = y[train_ix], y[test_ix]
  # define model
  model = get_model(n_inputs, n_outputs)
  # fit model
  model.fit(X_train, y_train, verbose=0, epochs=100)
  # make a prediction on the test set
  print("Fold: ", fold)
  yhat = model.predict(X_test)
  print(yhat)
  print(y_test)
  # round probabilities to class labels
  yhat = yhat.round()
  print(yhat)
  # calculate accuracy
  acc = accuracy_score(y_test, yhat)
  print(acc)
  # store result
  print('>%.3f' % acc)
  results.append(acc)
  fold = fold + 1

# # evaluate a model using repeated k-fold cross-validation
# def evaluate_model(X, y):
# 	results = list()
# 	n_inputs, n_outputs = X.shape[1], y.shape[1]
# 	# define evaluation procedure
# 	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# 	# enumerate folds
# 	for train_ix, test_ix in cv.split(X):
# 		# prepare data
# 		X_train, X_test = X[train_ix], X[test_ix]
# 		y_train, y_test = y[train_ix], y[test_ix]
# 		# define model
# 		model = get_model(n_inputs, n_outputs)
# 		# fit model
# 		model.fit(X_train, y_train, verbose=0, epochs=100)
# 		# make a prediction on the test set
# 		yhat = model.predict(X_test)
# 		# round probabilities to class labels
# 		yhat = yhat.round()
# 		# calculate accuracy
# 		acc = accuracy_score(y_test, yhat)
# 		# store result
# 		print('>%.3f' % acc)
# 		results.append(acc)
# 	return results

# # evaluate model
# results = evaluate_model(X, y)

# summarize performance
print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))