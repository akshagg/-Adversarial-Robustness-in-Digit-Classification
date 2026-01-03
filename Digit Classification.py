#MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA, PCA, NMF
import copy

rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)

def dataset_searcher(number_list, images, labels):
  indices = np.isin(labels, number_list)
  return images[indices], labels[indices]

def print_numbers(images, labels):
  _, axes = plt.subplots(1, len(labels), figsize=(10, 3))
  for ax, image, label in zip(axes, images, labels):
      ax.set_axis_off()
      ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
      ax.set_title("Label: %i" % label)
  plt.show()

class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher(class_numbers, images, labels)

# Sort images to match the order 2, 0, 8, 7, 5 exactly
ordered_images = []
ordered_labels = []
for num in class_numbers:
    idx = np.where(class_number_labels == num)[0][0]
    ordered_images.append(class_number_images[idx])
    ordered_labels.append(class_number_labels[idx])

#Part 2
print_numbers(ordered_images, ordered_labels)

model_1 = GaussianNB()

n_samples_train = len(X_train)
X_train_reshaped = X_train.reshape((n_samples_train, -1))
n_samples_test = len(X_test)
X_test_reshaped = X_test.reshape((n_samples_test, -1))

model_1.fit(X_train_reshaped, y_train)
model1_results = model_1.predict(X_test_reshaped)

#Part 3
def OverallAccuracy(results, actual_labels):
  correct = np.sum(results == actual_labels)
  return correct / len(actual_labels)

# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))

#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)

#Part 6
model_2 = KNeighborsClassifier(n_neighbors=5)
model_2.fit(X_train_reshaped, y_train)
model2_results = model_2.predict(X_test_reshaped)
Model2_Overall_Accuracy = OverallAccuracy(model2_results, y_test)
print("The overall results of the KNN model is " + str(Model2_Overall_Accuracy))

model_3 = MLPClassifier(random_state=1, max_iter=2000, hidden_layer_sizes=(200, 100))
model_3.fit(X_train_reshaped, y_train)
model3_results = model_3.predict(X_test_reshaped)
Model3_Overall_Accuracy = OverallAccuracy(model3_results, y_test)
print("The overall results of the MLP model is " + str(Model3_Overall_Accuracy))

#Part 8
#Poisoning
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison

#Part 9-11
X_train_poison_reshaped = X_train_poison.reshape((n_samples_train, -1))

model_1.fit(X_train_poison_reshaped, y_train)
poison_results1 = model_1.predict(X_test_reshaped)
print("Gaussian Accuracy (Poisoned): " + str(OverallAccuracy(poison_results1, y_test)))

model_2.fit(X_train_poison_reshaped, y_train)
poison_results2 = model_2.predict(X_test_reshaped)
print("KNN Accuracy (Poisoned): " + str(OverallAccuracy(poison_results2, y_test)))

model_3.fit(X_train_poison_reshaped, y_train)
poison_results3 = model_3.predict(X_test_reshaped)
print("MLP Accuracy (Poisoned): " + str(OverallAccuracy(poison_results3, y_test)))

#Part 12-13
# Denoise the poisoned training data
# Best Approach: PCA with decimal n_components (variance ratio).
# n_components=0.80 means "Select enough components to explain 80% of the variance."
# This allows the model to dynamically choose the best number of components based on information content.
pca = PCA(n_components=0.80, svd_solver='full')
X_train_denoised_reshaped = pca.fit_transform(X_train_poison_reshaped)
X_train_denoised = pca.inverse_transform(X_train_denoised_reshaped)

# CRITICAL FIX: Clip data to original pixel range (0-16).
# This removes negative values caused by PCA, which were destroying KNN accuracy.
X_train_denoised = np.clip(X_train_denoised, 0, 16)

#Part 14#MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA, PCA, NMF
import copy

rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)

def dataset_searcher(number_list, images, labels):
  indices = np.isin(labels, number_list)
  return images[indices], labels[indices]

def print_numbers(images, labels):
  _, axes = plt.subplots(1, len(labels), figsize=(10, 3))
  for ax, image, label in zip(axes, images, labels):
      ax.set_axis_off()
      ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
      ax.set_title("Label: %i" % label)
  plt.show()

class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher(class_numbers, images, labels)

ordered_images = []
ordered_labels = []
for num in class_numbers:
    idx = np.where(class_number_labels == num)[0][0]
    ordered_images.append(class_number_images[idx])
    ordered_labels.append(class_number_labels[idx])

#Part 2
print_numbers(ordered_images, ordered_labels)

model_1 = GaussianNB()

n_samples_train = len(X_train)
X_train_reshaped = X_train.reshape((n_samples_train, -1))
n_samples_test = len(X_test)
X_test_reshaped = X_test.reshape((n_samples_test, -1))

model_1.fit(X_train_reshaped, y_train)
model1_results = model_1.predict(X_test_reshaped)

#Part 3
def OverallAccuracy(results, actual_labels):
  correct = np.sum(results == actual_labels)
  return correct / len(actual_labels)

# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))

#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)

#Part 6
model_2 = KNeighborsClassifier(n_neighbors=5)
model_2.fit(X_train_reshaped, y_train)
model2_results = model_2.predict(X_test_reshaped)
Model2_Overall_Accuracy = OverallAccuracy(model2_results, y_test)
print("The overall results of the KNN model is " + str(Model2_Overall_Accuracy))

model_3 = MLPClassifier(random_state=1, max_iter=2000, hidden_layer_sizes=(200, 100))
model_3.fit(X_train_reshaped, y_train)
model3_results = model_3.predict(X_test_reshaped)
Model3_Overall_Accuracy = OverallAccuracy(model3_results, y_test)
print("The overall results of the MLP model is " + str(Model3_Overall_Accuracy))

#Part 8
#Poisoning
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison

#Part 9-11
X_train_poison_reshaped = X_train_poison.reshape((n_samples_train, -1))

model_1.fit(X_train_poison_reshaped, y_train)
poison_results1 = model_1.predict(X_test_reshaped)
Gaussian_Poison_Accuracy = OverallAccuracy(poison_results1, y_test)
print("Gaussian Accuracy (Poisoned): " + str(Gaussian_Poison_Accuracy))

model_2.fit(X_train_poison_reshaped, y_train)
poison_results2 = model_2.predict(X_test_reshaped)
KNN_Poison_Accuracy = OverallAccuracy(poison_results2, y_test)
print("KNN Accuracy (Poisoned): " + str(KNN_Poison_Accuracy))

model_3.fit(X_train_poison_reshaped, y_train)
poison_results3 = model_3.predict(X_test_reshaped)
MLP_Poison_Accuracy = OverallAccuracy(poison_results3, y_test)
print("MLP Accuracy (Poisoned): " + str(MLP_Poison_Accuracy))

#Part 12-13
# Denoise the poisoned training data
pca = PCA(n_components=0.80, svd_solver='full')
X_train_denoised_reshaped = pca.fit_transform(X_train_poison_reshaped)
X_train_denoised = pca.inverse_transform(X_train_denoised_reshaped)

X_train_denoised = np.clip(X_train_denoised, 0, 16)

#Part 14
model_1.fit(X_train_denoised, y_train)
denoised_results1 = model_1.predict(X_test_reshaped)
Gaussian_Denoised_Accuracy = OverallAccuracy(denoised_results1, y_test)
print("Gaussian Accuracy (Denoised): " + str(Gaussian_Denoised_Accuracy))

model_2 = KNeighborsClassifier(n_neighbors=5, weights='distance')
model_2.fit(X_train_denoised, y_train)
denoised_results2 = model_2.predict(X_test_reshaped)
KNN_Denoised_Accuracy = OverallAccuracy(denoised_results2, y_test)
print("KNN Accuracy (Denoised): " + str(KNN_Denoised_Accuracy))

model_3.fit(X_train_denoised, y_train)
denoised_results3 = model_3.predict(X_test_reshaped)
MLP_Denoised_Accuracy = OverallAccuracy(denoised_results3, y_test)
print("MLP Accuracy (Denoised): " + str(MLP_Denoised_Accuracy))

#Part 15
labels = ['Gaussian', 'KNN', 'MLP']
clean_scores = [Model1_Overall_Accuracy, Model2_Overall_Accuracy, Model3_Overall_Accuracy]
poison_scores = [Gaussian_Poison_Accuracy, KNN_Poison_Accuracy, MLP_Poison_Accuracy]
denoised_scores = [Gaussian_Denoised_Accuracy, KNN_Denoised_Accuracy, MLP_Denoised_Accuracy]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, clean_scores, width, label='Clean')
rects2 = ax.bar(x, poison_scores, width, label='Poisoned')
rects3 = ax.bar(x + width, denoised_scores, width, label='Denoised')

ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison: Clean vs Poisoned vs Denoised')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
model_1.fit(X_train_denoised, y_train)
denoised_results1 = model_1.predict(X_test_reshaped)
print("Gaussian Accuracy (Denoised): " + str(OverallAccuracy(denoised_results1, y_test)))

# Use 5 neighbors with distance weighting.
# Averaging neighbors smooths out noise, and distance weighting ensures good matches count more.
model_2 = KNeighborsClassifier(n_neighbors=5, weights='distance')
model_2.fit(X_train_denoised, y_train)
denoised_results2 = model_2.predict(X_test_reshaped)
print("KNN Accuracy (Denoised): " + str(OverallAccuracy(denoised_results2, y_test)))

model_3.fit(X_train_denoised, y_train)
denoised_results3 = model_3.predict(X_test_reshaped)
print("MLP Accuracy (Denoised): " + str(OverallAccuracy(denoised_results3, y_test)))