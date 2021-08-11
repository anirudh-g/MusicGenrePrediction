from sklearn import ensemble
from sklearn import linear_model
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
estimators = [
        ('rf', ensemble.RandomForestClassifier(random_state=42, class_weight='balanced', bootstrap=True)),
        ('lgbm', LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1)),
       # ('et', ensemble.ExtraTreesClassifier(class_weight='balanced', bootstrap=True, random_state=42))
    ]

stack_model = ensemble.StackingClassifier(
                estimators=estimators, 
                final_estimator=linear_model.LogisticRegression(
                penalty='l1', class_weight='balanced', solver='liblinear',
                 random_state=42), cv=5, verbose=2)
                        
