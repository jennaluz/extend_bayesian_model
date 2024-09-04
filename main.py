import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# User-Define Functions ========================================================

def save_heatmap(dataframe, filename, show=False):
    sns.heatmap(dataframe.corr())

    plt.tight_layout()
    plt.savefig(os.path.join('images', filename))

    if show:
        plt.show()

    plt.clf()


# Read in Data =================================================================
steel_data = pd.read_csv('steel_faults.csv')


# Dataset Structure ============================================================
pd.set_option('display.max_columns', None)

#print('Dataset Head')
#print(steel_data.head())
#print()


# Encode Fault Types ===========================================================
# These are found in the given dataset
fault_types = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
               'Dirtiness', 'Bumps', 'Other_Faults']

# Create a new column that will contain the encoded fault type
steel_data['Fault'] = 0     # All rows in this column will be 0

for i in range(0, len(fault_types)):
    # Find all indices of the fault type
    fault_indices = steel_data.loc[steel_data[fault_types[i]] == 1].index.tolist()

    # Store the encoded fault type into the corresponding indices
    steel_data.loc[fault_indices, 'Fault'] = i + 1


# Initialize Models ============================================================
# Model with full set of features
model_original = GaussianNB()
model_original_drop_features = fault_types + ['Fault']
model_original_inputs = steel_data.drop(model_original_drop_features, axis = 1)
save_heatmap(model_original_inputs, 'model_original_heatmap.png')

# Model with correlated features
model_1 = GaussianNB()
model_1_features = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum',
            'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter']
model_1_inputs = steel_data[model_1_features]
save_heatmap(model_1_inputs, 'model_1_heatmap.png')

# Model with uncorrelated features
model_2 = GaussianNB()
model_2_drop_features = fault_types + ['Fault', 'X_Minimum', 'Y_Minimum',
                                   'X_Perimeter', 'Y_Perimeter',
                                   'Sum_of_Luminosity', 'Log_X_Index',
                                   'Log_Y_Index', 'Luminosity_Index',
                                   'Maximum_of_Luminosity', 'LogOfAreas',
                                   'Outside_Global_Index', 'Orientation_Index']
model_2_inputs = steel_data.drop(model_2_drop_features, axis = 1)
save_heatmap(model_2_inputs, 'model_2_heatmap.png')

targets = steel_data['Fault']

training_inputs = pd.DataFrame()
training_targets = pd.DataFrame()
test_inputs = pd.DataFrame()
test_targets = pd.DataFrame()

iterations = 500
# Row 0 records the mean average of that iteration for Model 1
# Row 1 records the mean average of that iteration for Model 2
# Row 2 records the mean average of that iteration for Model 3
model_accuracy = np.zeros((4, iterations))

# Train the Models =============================================================
for i in range(iterations):
    state_seed = random.randrange(100)

    # Original Model
    training_inputs, test_inputs, training_targets, test_targets = \
            train_test_split(model_original_inputs, targets, test_size = 0.1, random_state = state_seed)
    model_original.fit(training_inputs, training_targets)
    model_accuracy[0, i] = model_original.score(test_inputs, test_targets)


    # Model 1
    training_inputs, test_inputs, training_targets, test_targets = \
            train_test_split(model_1_inputs, targets, test_size = 0.1, random_state = state_seed)
    model_1.fit(training_inputs, training_targets)
    model_accuracy[1, i] = model_1.score(test_inputs, test_targets)


    # Model 2
    training_inputs, test_inputs, training_targets, test_targets = \
            train_test_split(model_2_inputs, targets, test_size = 0.1, random_state = state_seed)
    model_2.fit(training_inputs, training_targets)
    model_accuracy[2, i] = model_2.score(test_inputs, test_targets)


# Model Comparison =============================================================
print('===== Accuracy Table =====')
print('Original Model\t\t\tModel 1\t\t\tModel 2')
print(f'{np.average(model_accuracy[0])}\t{np.average(model_accuracy[1])}\t{np.average(model_accuracy[2])}\t')
print()


# Model 2 Test Splits ==========================================================
# Row 0 will record the mean accuracy of a 5% test size
# Row 1 will record the mean accuracy of a 20% test size
model_2_accuracy = np.zeros((2, iterations))

for i in range(iterations):
    # Use 5% as a test size
    training_inputs, test_inputs, training_targets, test_targets = \
            train_test_split(model_2_inputs, targets, test_size = 0.05)
    model_2.fit(training_inputs, training_targets)
    model_2_accuracy[0, i] = model_2.score(test_inputs, test_targets)

    # Use 20% as a test size
    training_inputs, test_inputs, training_targets, test_targets = \
            train_test_split(model_2_inputs, targets, test_size = 0.2)
    model_2.fit(training_inputs, training_targets)
    model_2_accuracy[1, i] = model_2.score(test_inputs, test_targets)

# Test Size Comparison
print('===== Test Size Table =====')
print('5%\t\t\t20%')
print(f'{np.average(model_2_accuracy[0])}\t{np.average(model_2_accuracy[1])}')
print()


# Model 2 Predictions ==========================================================
training_inputs, test_inputs, training_targets, test_targets = \
        train_test_split(model_2_inputs, targets, test_size = 0.2)
model_2.fit(training_inputs, training_targets)

predictions = model_2.predict(test_inputs)
targets = test_targets.to_numpy()
true_positives = np.zeros(7)
false_negatives = np.zeros(7)

for i in range(len(targets)):
    fault_type = targets[i] - 1

    if predictions[i] == targets[i]:
        true_positives[fault_type] = true_positives[fault_type] + 1
    else:
        false_negatives[fault_type] = false_negatives[fault_type] + 1

true_positive_rate = true_positives / (true_positives + false_negatives)

# Prediction Statistics
print('===== Prediction Table =====')
print('\tPastry\t\tZ Scratch\tK Scratch\tStains\t\tDirtiness\tBumps\t\tOther Faults')
print(f'TP\t{true_positives[0]}\t\t{true_positives[1]}\t\t{true_positives[2]}\t\t{true_positives[3]}\t\t{true_positives[4]}\t\t{true_positives[5]}\t\t{true_positives[6]}')
print(f'FN\t{false_negatives[0]}\t\t{false_negatives[1]}\t\t{false_negatives[2]}\t\t{false_negatives[3]}\t\t{false_negatives[4]}\t\t{false_negatives[5]}\t\t{false_negatives[6]}')
print(f'TPR\t{true_positive_rate[0]:.2f}\t\t{true_positive_rate[1]:.2f}\t\t{true_positive_rate[2]:.2f}\t\t{true_positive_rate[3]:.2f}\t\t{true_positive_rate[4]:.2f}\t\t{true_positive_rate[5]:.2f}\t\t{true_positive_rate[6]:.2f}')
