import numpy as np
from scipy.io import loadmat
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data from .mat files
def load_data(files):
    data_list = []
    labels = []
    for file, label in files:
        mat_data = loadmat(file)
        data = mat_data['all_data']  # Assuming the data is stored under the key 'all_data' in .mat file
        data_list.append(data)
        labels.extend([label] * data.shape[0])
    return data_list, np.array(labels)

# List of .mat files and their corresponding labels
files = [
    ('t0.mat', 'no_fault'),
    ('ta1.mat', 'fault_1'),
    ('ta2.mat', 'fault_2'),
    ('ta3.mat', 'fault_3'),
    ('ta4.mat', 'fault_4'),
    ('tb1.mat', 'fault_5'),
    ('tb2.mat', 'fault_6'),
    ('tb3.mat', 'fault_7'),
    ('tb4.mat', 'fault_8'),
    ('tc1.mat', 'fault_9'),
    ('tc2.mat', 'fault_10'),
    ('tc3.mat', 'fault_11'),
    ('tc4.mat', 'fault_12'),
    # Add more files and corresponding labels as needed
]

# Load the data
data_list, labels = load_data(files)

# Concatenate all data
all_data = np.concatenate(data_list)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(all_data)

# Encode the labels into numeric format
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels_encoded, test_size=0.2, random_state=42)

# Initialize the Extra Trees classifier
et = ExtraTreesClassifier(n_estimators=100)

# Train the classifier
et.fit(X_train, y_train)

# Predict fault conditions
predictions = et.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Extra Trees Classifier Accuracy SPWM:", accuracy)

# Create confusion matrix
cm = confusion_matrix(y_test, predictions)

# Plot confusion matrix
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=100)
plt.show()
