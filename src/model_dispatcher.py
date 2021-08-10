from sklearn import ensemble

estimators = [
        ('bagging', ensemble.RandomForestClassifier(random_state=42, class_weight='balanced', bootstrap=True)), 
        ('boosting', ensemble.GradientBoostingClassifier(random_state=42))
    ]

stack_model = ensemble.StackingClassifier(
                    estimators=estimators, 
                    final_estimator=ensemble.ExtraTreesClassifier(
                        random_state=42, class_weight='balanced', bootstrap=True), cv=5, verbose=2)
                        
