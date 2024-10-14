import random
import math
import numpy as np
import pandas as pd
from mlp import MLP
from sklearn.metrics import roc_curve, precision_recall_curve, auc  # for computing the AUC and AUPRC
from sklearn.utils import resample
from scipy.stats import sem, t  # for confidence intervals

# data preprocessing - fetching data and creating samples
data = []
with open('./data_banknote_authentication_outliers.data') as f:
    for line in f.readlines():
        data.append(np.fromstring(line.rstrip('\n'), dtype=float, sep=','))
random.shuffle(data)
samples = pd.DataFrame(data, columns=["Variance", "Skewness", "Curtosis", "Entropy", "Class"])

# transfer labels to another dataframe
labels = samples["Class"]
samples = samples.drop("Class", axis=1)

# feature scaling: min-max method
def minMax(column):
    minVal = column.min()
    maxVal = column.max()
    scaled = (column - minVal) / (maxVal - minVal)
    return scaled
samples = samples.apply(minMax)

# 10-fold cross validation
print("Starting 10-fold cross-validation")
samples = samples.to_numpy()
labels = labels.to_numpy()
fold_size = math.ceil(len(samples)/10)

# initialize performance metrics
accuracy = []
precision = []
recall = []
f_score = []
AUC = []
AUPRC = []

for fold in range(10):
    # get training and test subsets for the iteration
    start = fold * fold_size
    if fold < 9:
        end = (fold + 1) * fold_size
    else:
        end = len(samples)
    test_samples = samples[start:end]
    test_labels = labels[start:end]
    train_samples = np.concatenate([samples[:start], samples[end:]])
    train_labels = np.concatenate([labels[:start], labels[end:]])

    # create the network
    network = MLP([4, 4, 1])

    # train the network with the training subset (set fifth argument to 1 to monitor cost progression across epochs)
    network.train(train_samples, train_labels, 50, 0.1)

    # initialize values necessary for performance evaluation
    correct_predictions = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    predictions = []  # for AUC

    # evaluate the trained network with the test subset
    for j in range(len(test_samples)):
        x = test_samples[j]
        y = test_labels[j]
        y_hat = network.classify(x)
        # get prediction
        if y_hat < 0.5:
            y_hat = 0
        else:
            y_hat = 1
        predictions.append(y_hat)

        # compare prediction to true labels
        if y_hat == y:
            correct_predictions = correct_predictions + 1
            if y_hat == 0:
                true_positives = true_positives + 1
        else:
            if y_hat == 0:
                false_positives = false_positives + 1
            else:
                false_negatives = false_negatives + 1

    # update performance metrics
    accuracy.append(correct_predictions / len(test_samples))
    pr = true_positives / (true_positives + false_positives)
    rc = true_positives / (true_positives + false_negatives)
    precision.append(pr)
    recall.append(rc)
    f_score.append((2*pr*rc)/(pr + rc))
    fpr, tpr, thresholds = roc_curve(test_labels, predictions)
    AUC.append(auc(fpr, tpr))
    pr_curve, rc_curve, thresholds = precision_recall_curve(test_labels, predictions)
    AUPRC.append(auc(rc_curve, pr_curve))

    print("STEP " + str(fold+1) + "/10 COMPLETE")

# average out the performance metrics
accuracy_mean = np.mean(np.array(accuracy))
precision_mean = np.mean(np.array(precision))
recall_mean = np.mean(np.array(recall))
f_score_mean = np.mean(np.array(f_score))
AUC_mean = np.mean(np.array(AUC))
AUPRC_mean = np.mean(np.array(AUPRC))

# confidence intervals (bootstrapping)
def confidence_intervals(metric_values, confidence_level=0.95, num_samples=1000):
    bootstrapped_metrics = np.empty(num_samples)
    for i in range(num_samples):
        sample = resample(metric_values)
        bootstrapped_metrics[i] = np.mean(sample)
    confidence_interval = t.interval(confidence_level, len(bootstrapped_metrics) - 1, loc=np.mean(bootstrapped_metrics), scale=sem(bootstrapped_metrics))
    return confidence_interval

# Final output
print("\nAccuracy: " + str(accuracy_mean) + "  Confidence interval: " + str(confidence_intervals(accuracy)))
print("Precision: " + str(precision_mean) + " Confidence interval: " + str(confidence_intervals(precision)))
print("Recall: " + str(recall_mean) + "    Confidence interval: " + str(confidence_intervals(recall)))
print("F-Score: " + str(f_score_mean) + "   Confidence interval: " + str(confidence_intervals(f_score)))
print("AUC: " + str(AUC_mean) + "       Confidence interval: " + str(confidence_intervals(AUC)))
print("AUPRC: " + str(AUPRC_mean) + "     Confidence interval: " + str(confidence_intervals(AUPRC)))
