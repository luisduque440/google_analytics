import pandas as pd
import numpy as np
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
