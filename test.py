import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # load in the test data
    test = pd.read_csv('test.csv', header=None)
    # test = test.interpolate(method='linear', axis=1)
    test.columns = range(0, 120, 5)

    print (test)
    # load in the classifier
    clf = pickle.load(open('train.pickle', 'rb'))
    sc = StandardScaler()

    # feature extraction
    tmax_tmin = []
    max_min = []
    derivative = []
    rolling_mean = []
    window_mean = []
    labels = []

    window1 = test[[0, 5, 10, 15, 20, 25]]
    window2 = test[[30, 35, 40, 45, 50, 55]]
    window3 = test[[60, 65, 70, 75, 80, 85]]
    window4 = test[[90, 95, 100, 105, 110, 115]]

    for i in range(len(test)):
        # feature 1 - difference in cgm max and min time
        testMaxIndex = test.iloc[i].loc[test.iloc[i] == test.iloc[i].max()].index.tolist()[0]
        testMinIndex = test.iloc[i].loc[test.iloc[i] == test.iloc[i].min()].index.tolist()[0]
        tmax_tmin.append(testMaxIndex - testMinIndex)

        # feature 2 - difference in cgm max and min
        test_diff_cgm = test.iloc[i].max() - test.iloc[i].min()
        max_min.append(test_diff_cgm)

        # feature 3 - max of derivative cgm
        derivative.append((test.iloc[i].diff() / 5).max())

        # feature 4 - rolling mean
        rolling_mean.append(test.iloc[i].rolling(window=5).mean().mean())

        # feature 5 - window mean
        window_mean.append((window1.iloc[i].mean() + window2.iloc[i].mean() + window3.iloc[i].mean() +
                            window4.iloc[i].mean()) / 5)

    featureMatrix = pd.DataFrame(columns=[1, 2, 3, 4, 5], index=range(len(test)))
    featureMatrix[1] = tmax_tmin
    featureMatrix[2] = max_min
    featureMatrix[3] = derivative
    featureMatrix[4] = rolling_mean
    featureMatrix[5] = window_mean

    # scale the feature matrix and test the model
    test_features = sc.fit_transform(featureMatrix)
    test = sc.transform(test_features)
    test_predictions = clf.predict(test)

    # save results in csv
    results = pd.DataFrame(data=test_predictions)
    results.to_csv('Result.csv', header=None, index=None)
