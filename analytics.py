import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

def get_validation_report(model, X_train, y_train, X_validation, y_validation):
    y_predicted = model.predict(X_validation)
    v_mse, v_benchmarks_mse, v_improvement = evaluate_model(y_validation, y_predicted)
    y_predicted = model.predict(X_train)
    t_mse, t_benchmarks_mse, t_improvement = evaluate_model(y_train, y_predicted)
    A= pd.DataFrame({
            "train_mse":[t_mse], 
            "train_benchamrk":[t_benchmarks_mse], 
            "train_improvement":[t_improvement],
            "validation_mse":[v_mse], 
            "validation_benchmark":[v_benchmarks_mse],
            "validation_improvement":[v_improvement]
            })
    return A

    
def evaluate_model(y_true, y_predicted):
    y_all_zeros = [0]*len(y_true)
    mse = mean_squared_error(y_true, y_predicted)
    benchmark_mse = mean_squared_error(y_true, y_all_zeros)
    improvement = 100*(mse-benchmark_mse)/benchmark_mse
    return mse, benchmark_mse, improvement



def plot_roc_curve(y_true, y_predicted_probability, title=""):
    fpr, tpr, thresholds = roc_curve(y_true, y_predicted_probability)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='auc = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.title(title, fontsize=25)
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.legend(loc="lower right", fontsize=20)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.show()
    
    
def plot_precision_recall_curve(y_true, y_predicted_probability, title=""):
    precision, recall, threshold = precision_recall_curve(y_true, y_predicted_probability)
    average_precision = average_precision_score(y_true, y_predicted_probability)
    coin_flip_precision = 1.0*sum(y_true)/len(y_true)
    plt.figure(figsize=(10,10))
    plt.plot(recall, precision,  color='darkorange', lw=2, label='ap = %0.3f' %average_precision)
    plt.plot([0, 1], [coin_flip_precision, coin_flip_precision], color='black', lw=2, linestyle='--')
    plt.title(title, fontsize=25)
    plt.xlim([0.0, 1.0])
    plt.ylim([coin_flip_precision - 0.1, coin_flip_precision + 0.5])
    plt.grid()
    plt.xlabel('Recall', fontsize=25)
    plt.ylabel('Precision', fontsize=25)
    plt.legend(loc="upper right", fontsize=20)

    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.show()
    return