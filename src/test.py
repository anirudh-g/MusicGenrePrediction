import pandas as pd
from sklearn import preprocessing
import config
import joblib

test_data = pd.read_csv(config.TEST_FINAL)
sub_temp = pd.read_csv(config.submission)


trained_model = joblib.load(config.TRAINED_MODEL)
finalpreds = pd.DataFrame(trained_model.predict_proba(test_data), columns=list(sub_temp.columns))
print('Finished running the model. Check data->submission.csv for the output')
finalpreds.to_csv(r"C:\Users\ganir\iCloudDrive\ML Competitions\MachineHack\MusicGenrePrediction\data\submission.csv", index=False)