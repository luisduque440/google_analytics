clf = RandomForestClassifier(class_weight='balanced')
param_grid = {
    'min_samples_leaf' : [2, 5, 10, 20], # Best:
    'max_depth': [2, 10, 20], # Best:
    'n_estimators': [5, 20, 100, 200] # Best:      
}

search = GridSearchCV(clf, param_grid)
search.fit(X_train, y_train)

y_predicted_probability = search.predict_proba(X_validation)[:,1]
plot_roc_curve(y_validation, y_predicted_probability, title="ROC in test set RF")
plot_precision_recall_curve(y_validation, y_predicted_probability)

all_features =pd.DataFrame({'feature': X_train.columns, 'importance': search.best_estimator_.feature_importances_})
all_features = all_features.sort_values(by=['importance'], ascending=False).set_index('feature')
all_features[all_features.importance>0].plot.bar()

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)


###############################

clf = XGBClassifier()
param_grid = {
    'max_depth':[3, 8, 20],    
    'n_estimators': [3, 50, 100], 
    'learning_rate' : [0.1], 
    'min_child_weight' : [2, 100], 
    'reg_lambda': [1, 20, 30] 
}

search = GridSearchCV(clf, param_grid)
search.fit(X_train, y_train)

y_predicted_probability = search.predict_proba(X_validation)[:,1]
plot_roc_curve(y_validation, y_predicted_probability, title="ROC in test set RF")
plot_precision_recall_curve(y_validation, y_predicted_probability)

all_features =pd.DataFrame({'feature': X_train.columns, 'importance': search.best_estimator_.feature_importances_})
all_features = all_features.sort_values(by=['importance'], ascending=False).set_index('feature')
all_features[all_features.importance>0].plot.bar()

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)