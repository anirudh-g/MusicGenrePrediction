import pandas as pd
from sklearn import model_selection
import config

if __name__ == "__main__":
    train = pd.read_csv(config.TRAINING_FILE)
    test = pd.read_csv(config.TEST_FILE)

    train.drop_duplicates(inplace=True)

    train['new4'] = train.groupby(['Track Name'])['acousticness'].transform('sum')
    train['new5'] = train.groupby(['Artist Name'])['Track Name'].transform('count')
    train['new6'] = train.groupby(['Artist Name', 'Track Name'])['Popularity'].transform('mean')
    train['new7'] = train.groupby(['Artist Name', 'mode'])['Popularity'].transform('min')
    train['new8'] = train.groupby(['Track Name', 'time_signature'])['tempo'].transform('mean')
    train['new9'] = train.groupby(['Artist Name'])['valence'].transform('mean')
    train['new10'] = train.groupby(['Track Name'])['duration_in min/ms'].transform('mean')

    test['new4'] = test.groupby(['Track Name'])['acousticness'].transform('sum')
    test['new5'] = test.groupby(['Artist Name'])['Track Name'].transform('count')
    test['new6'] = test.groupby(['Artist Name', 'Track Name'])['Popularity'].transform('mean')
    test['new7'] = test.groupby(['Artist Name', 'mode'])['Popularity'].transform('min')
    test['new8'] = test.groupby(['Track Name', 'time_signature'])['tempo'].transform('mean')
    test['new9'] = test.groupby(['Artist Name'])['valence'].transform('mean')
    test['new10'] = test.groupby(['Track Name'])['duration_in min/ms'].transform('mean')


    y_train = train[['Class']]
    X_train = train.drop(['Class', 'Track Name', 'Artist Name'], axis=1)
    test = test.drop(['Track Name', 'Artist Name'], axis=1)
    
    X_train.to_csv(r'C:\Users\ganir\iCloudDrive\ML Competitions\MachineHack\MusicGenrePrediction\data\X_train.csv', index=False)
    y_train.to_csv(r'C:\Users\ganir\iCloudDrive\ML Competitions\MachineHack\MusicGenrePrediction\data\y_train.csv', index=False)
    test.to_csv(r'C:\Users\ganir\iCloudDrive\ML Competitions\MachineHack\MusicGenrePrediction\data\test_final.csv', index=False)
