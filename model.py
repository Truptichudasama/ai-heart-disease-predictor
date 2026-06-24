
from train_test_split import * 


def fit_and_evaluate_model(x_train, x_test, y_train, y_test,xgb):
    xgb.fit(x_train, y_train)
    xgb_predict = xgb.predict(x_test)
    xgb_conf_matrix = confusion_matrix(y_test, xgb_predict)
    xgb_acc_score = accuracy_score(y_test, xgb_predict)
    print("confussion matrix")
    print(xgb_conf_matrix)
    print("\n")
    print("Accuracy of XGBoost:",xgb_acc_score*100,'\n')
    print(classification_report(y_test,xgb_predict))
    return xgb


print("********************************************************************")
print("++++++++++++++++++First evalution+++++++++++++++++++++++++")
xgb =  XGBClassifier()
model = fit_and_evaluate_model(x_train, x_test, y_train, y_test,xgb)



importances = pd.DataFrame(model.feature_importances_) # calculated gain
importances['features'] = features
importances.columns = ['importance','feature']
importances.sort_values(by = 'importance', ascending= True,inplace=True)

model.fit(x_train, y_train)

importance = model.feature_importances_
features = x_train.columns

# Sort features by importance
sorted_idx = np.argsort(importance)[::-1]
sorted_features = features[sorted_idx]

best_score = 0
best_n = 0

for i in range(5, len(features), 5):   # try top 5,10,15...
    selected = sorted_features[:i]
    
    model.fit(x_train[selected], y_train)
    preds = model.predict_proba(x_test[selected])[:,1]
    
    score = roc_auc_score(y_test, preds)
    
    print(f"Top {i} features → AUC: {score}")
    
    if score > best_score:
        best_score = score
        best_n = i

print(f"\nBest number of features: {best_n}")
best_features = sorted_features[:30]
#Best features saved to best_features.csv
# Convert to DataFrame
df_features = pd.DataFrame(best_features, columns=['Feature_Name'])

# Save to CSV
df_features.to_csv('best_features.csv', index=False)

print("Best features saved to best_features.csv")

for i, f in enumerate(best_features, 1):
    print(f"{i}. {f}")



x = df[best_features]
y = df['HeartDisease']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.67, shuffle = False, random_state = 0)


print("********************************************************************")
print("++++++++++++++++++Second evalution+++++++++++++++++++++++++")
xgb =  XGBClassifier()
model = fit_and_evaluate_model(x_train, x_test, y_train, y_test,xgb)

#with best perameter 

new_xgb_model = XGBClassifier(
    n_estimators=550,
    max_depth=10,
    learning_rate=0.10849394135707167,
    subsample=0.619635497139044,
    colsample_bytree=0.669615416953633,
    min_child_weight=3,
    reg_alpha=0.5433096858197747,
    reg_lambda=0.4952639725801321,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
    eval_metric="auc"
)
new_xgb_model.fit(x, y)

print("**************************")
print("After hyper tuning")
preds = new_xgb_model.predict_proba(x_test)[:, 1]
auc = roc_auc_score(y_test, preds)
print(auc)

# Probabilities → for AUC
preds_proba = new_xgb_model.predict_proba(x_test)[:, 1]

# Class labels → for accuracy
preds = new_xgb_model.predict(x_test)

# Metrics
auc = roc_auc_score(y_test, preds_proba)
accuracy = accuracy_score(y_test, preds)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

print("Final report")
print("AUC:", auc)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, preds))


# Save the model
joblib.dump(new_xgb_model, "model_classifier.pkl")
