This repo is part of the "Music Genre Prediction" weekend hackathon hosted on MachineHack

**This solution was placed at Rank 44 (out of 329 participants)**

Main Solution structure: 
--> notebooks
      - eda.ipynb : Contains basic eda and exploration tasks
--> src 
      - clean_and_extract.py : Data Cleaning and feature extraction to create the final dataset used for building the ML pipeline
      - config.py : Input and output paths (Points to my local machine)
      - dockerfile : Contains Docker instructions to build the docker image (under construction)
      - model_dispatcher.py : Contains the ML model(s) to be used by the train.py file
      - requirements.txt : API or framework requirements of this project
      - train.py : Contains the full model pipeline and the training script to train the model
      - test.py : Uses the trained model to produce the output/submission file


Brief overview of the solution
--> There are 2 key parts to the solution:
      - Data Cleaning & Feature Extraction : The duplicate data points were removed and new features were extracted based on the observation from EDA. The categorical features
                                             were transformed to numerical features by grouping them with relevant numerical features Then as part of the ML 
                                             pipeline, the outliers are treated by taking the values between 5 to 95 perctile of the data. Then MICE imputer was used to 
                                             impute the missing values in the data. (For testing purpose - The data was scaled and labelencoded)
       
      - ML Model: A bagging based classifer Random Forest and a boosting based classifier XGBoost were Stacked by using the Stacking Classifier. Logistic regression with
                  L1 penalty (which acts as an feature selector) was used as the final estimator  
