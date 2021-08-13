from sklearn import ensemble
from sklearn import linear_model
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
estimators = [
        ('rf', ensemble.RandomForestClassifier(random_state=42, class_weight='balanced', bootstrap=True)),
        ('xgb', XGBClassifier(random_state=42))
    ]

stack_model = ensemble.StackingClassifier(
                estimators=estimators, 
                final_estimator=linear_model.LogisticRegression(
                penalty='l1', class_weight='balanced', solver='liblinear',
                 random_state=42), cv=5, verbose=2)
                        
