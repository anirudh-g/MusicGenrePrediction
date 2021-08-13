import pandas as pd
import numpy as np
from sklearn import pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute
from sklearn import preprocessing
from sklearn import compose
from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble
import joblib
import os
import config
import model_dispatcher


def run():
    ''' The function consists of the model pipeline that will be used to run the input model'''
    
    numeric_transformer = pipeline.Pipeline(steps=[
        ('outliers', preprocessing.RobustScaler(quantile_range=(5,95))),
        ('imputer', impute.IterativeImputer(random_state=42, max_iter=100)), 
        ('scale', preprocessing.MinMaxScaler())
    ])

    categorical_tranformer = pipeline.Pipeline(steps=[
        ('encode', preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=99999))
    ])

    preprocess_pipeline = compose.ColumnTransformer(
        transformers=
        [
            ('num', numeric_transformer, compose.make_column_selector(dtype_exclude='object')), 
            ('cat', categorical_tranformer, compose.make_column_selector(dtype_include='object')), 
        ],

        remainder='passthrough')
    
    estimators = [
        ('bagging', ensemble.RandomForestClassifier(random_state=42, class_weight='balanced', bootstrap=True)), 
        ('boosting', ensemble.GradientBoostingClassifier(random_state=42))
    ]

    clf = pipeline.Pipeline(steps=[
        ('preprocessor', preprocess_pipeline),      
        ('model', model_dispatcher.stack_model),
        ]
    )

    X_train = pd.read_csv(config.X_train)
    y_train = pd.read_csv(config.y_train)

    #print(model_selection.cross_val_score(clf, X_train, y_train.values.ravel(), cv=5, verbose=1, 
    #scoring=metrics.make_scorer(metrics.log_loss, greater_is_better=True, 
    #needs_proba=True, labels=sorted(np.unique(y_train.values.ravel())))))

    clf.fit(X_train, y_train.values.ravel())

    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"trained_model.bin"))


if __name__ == "__main__":
    run()