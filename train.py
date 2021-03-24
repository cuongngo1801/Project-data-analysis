import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def run_script():
    insulin1 = pd.read_csv('InsulinData.csv', low_memory=False,
                           usecols=['Index', 'Date', 'Time', 'BWZ Carb Input (grams)'])
    insulin2 = pd.read_csv('Insulin_patient2.csv', low_memory=False,
                           usecols=['Index', 'Date', 'Time', 'BWZ Carb Input (grams)'])
    cgm1 = pd.read_csv('CGMData.csv', low_memory=False, usecols=['Index', 'Date', 'Time', 'Sensor Glucose (mg/dL)'])
    cgm2 = pd.read_csv('CGM_patient2.csv', low_memory=False,
                       usecols=['Index', 'Date', 'Time', 'Sensor Glucose (mg/dL)'])

    # changing data types
    cgm1['DateTime'] = cgm1['Date'] + ' ' + cgm1['Time']

    cgm1['DateTime'] = pd.to_datetime(cgm1['DateTime'], format="%m/%d/%Y %H:%M:%S")

    cgm1['Time'] = pd.to_datetime(cgm1['Time'], format="%H:%M:%S").dt.time
    cgm2['Date'] = pd.to_datetime(cgm2.Date)
    cgm2['Date'] = cgm2['Date'].dt.strftime('%m/%d/%Y')
    cgm2['DateTime'] = cgm2['Date'] + ' ' + cgm2['Time']
    # print(cgm2['DateTime'])
    cgm2['DateTime'] = pd.to_datetime(cgm2['DateTime'], format="%m/%d/%Y %H:%M:%S")
    # print(cgm2['DateTime'])
    cgm2['Time'] = pd.to_datetime(cgm2['Time'], format="%H:%M:%S").dt.time
    insulin1['DateTime'] = insulin1['Date'] + ' ' + insulin1['Time']
    insulin1['DateTime'] = pd.to_datetime(insulin1['DateTime'], format="%m/%d/%Y %H:%M:%S")
    insulin1['Time'] = pd.to_datetime(insulin1['Time'], format="%H:%M:%S").dt.time
    insulin2['Date'] = pd.to_datetime(insulin2.Date)
    insulin2['Date'] = insulin2['Date'].dt.strftime('%m/%d/%Y')
    insulin2['DateTime'] = insulin2['Date'] + ' ' + insulin2['Time']
    # print(insulin2['DateTime'] )
    insulin2['DateTime'] = pd.to_datetime(insulin2['DateTime'], format="%m/%d/%Y %H:%M:%S")
    insulin2['Time'] = pd.to_datetime(insulin2['Time'], format="%H:%M:%S").dt.time

    # get the indices of starting meal data and no meal data
    meals1 = \
        insulin1.loc[(insulin1['BWZ Carb Input (grams)'].isna() == False) & (insulin1['BWZ Carb Input (grams)'] != 0)][
            ['Index', 'DateTime']]
    meals2 = \
        insulin2.loc[(insulin2['BWZ Carb Input (grams)'].isna() == False) & (insulin2['BWZ Carb Input (grams)'] != 0)][
            ['Index', 'DateTime']]
    meals1['DiffInMeals'] = meals1['DateTime'].diff(-1).to_frame()
    meals2['DiffInMeals'] = meals2['DateTime'].diff(-1).to_frame()
    mealStartTime1 = meals1.loc[meals1['DiffInMeals'] > '02:00:00']
    mealStartTime2 = meals2.loc[meals2['DiffInMeals'] > '02:00:00']
    noMealStartTime1 = meals1.loc[meals1['DiffInMeals'] >= '04:00:00']
    noMealStartTime2 = meals2.loc[meals2['DiffInMeals'] >= '04:00:00']
    # print(mealStartTime1)
    # print(noMealStartTime1)
    get_extraction(mealStartTime1, mealStartTime2, cgm1, cgm2, noMealStartTime1, noMealStartTime2)


def get_extraction(mealStartTime1, mealStartTime2, cgm1, cgm2, noMealStartTime1, noMealStartTime2):
    # extract the meal time from cgm and get the glucose reading
    mealData1Cols = list(range(-30, 120, 5))
    # print(mealData1Cols);
    mealData1 = pd.DataFrame(columns=mealData1Cols, index=range(len(mealStartTime1)))
    for i in range(len(mealStartTime1) - 2):
        mealStartTemp = cgm1.loc[cgm1['DateTime'] >= mealStartTime1.iloc[i]['DateTime']]
        mealStartIndex = len(mealStartTemp) - 1
        preMealStartIndex = mealStartIndex + 6
        postMealStartIndex = mealStartIndex - 23
        meal_i = cgm1.iloc[postMealStartIndex:preMealStartIndex + 1, :]['Sensor Glucose (mg/dL)'].iloc[::-1].to_numpy()
        mealData1.iloc[i] = meal_i

    mealData1.dropna(inplace=True)
    print(mealData1)
    mealData2 = pd.DataFrame(columns=mealData1Cols, index=range(len(mealStartTime2)))
    for i in range(len(mealStartTime2) - 2):
        mealStartTemp = cgm2.loc[cgm2['DateTime'] >= mealStartTime2.iloc[i]['DateTime']]
        mealStartIndex = len(mealStartTemp) - 1
        preMealStartIndex = mealStartIndex + 6
        postMealStartIndex = mealStartIndex - 23
        meal_i = cgm2.iloc[postMealStartIndex:preMealStartIndex + 1, :]['Sensor Glucose (mg/dL)'].iloc[::-1].to_numpy()
        mealData2.iloc[i] = meal_i
    mealData2.dropna(inplace=True)
    mealData = mealData1.append(mealData2)
    mealData.reset_index(inplace=True)
    mealData = mealData.drop(['index'], axis=1)

    noMealData1Cols = list(range(0, 120, 5))
    # print(noMealData1Cols)
    noMealData1 = pd.DataFrame(columns=noMealData1Cols, index=range(len(noMealStartTime1)))
    # print(noMealData1)
    for i in range(len(noMealStartTime1) - 2):
        noMealStartTemp = cgm1.loc[cgm1['DateTime'] >= noMealStartTime1.iloc[i]['DateTime']]
        noMealStartIndex = len(noMealStartTemp) - 1
        postNoMealStartIndex = noMealStartIndex - 23
        noMeal_i = cgm1.iloc[postNoMealStartIndex:noMealStartIndex + 1, :]['Sensor Glucose (mg/dL)'].iloc[
                   ::-1].to_numpy()
        noMealData1.iloc[i] = noMeal_i
    noMealData1.dropna(inplace=True)

    noMealData2 = pd.DataFrame(columns=noMealData1Cols, index=range(len(noMealStartTime2)))
    for i in range(len(noMealStartTime2) - 2):
        noMealStartTemp = cgm2.loc[cgm2['DateTime'] >= noMealStartTime2.iloc[i]['DateTime']]
        noMealStartIndex = len(noMealStartTemp) - 1
        postNoMealStartIndex = noMealStartIndex - 23
        noMeal_i = cgm2.iloc[postNoMealStartIndex:noMealStartIndex + 1, :]['Sensor Glucose (mg/dL)'].iloc[
                   ::-1].to_numpy()
        noMealData2.iloc[i] = noMeal_i

    noMealData2.dropna(inplace=True)
    noMealData = noMealData1.append(noMealData2)
    noMealData.reset_index(inplace=True)
    noMealData = noMealData.drop(['index'], axis=1)
    # print(mealData)
    # print(noMealData)
    get_feature(mealData, noMealData)


def get_feature(mealData, noMealData):
    tmax_tmin = []
    max_min = []
    derivative = []
    rolling_mean = []
    window_mean = []
    labels = []
    meal_window1 = mealData[[-30, -25, -20, -15, -10, -5]]
    meal_window2 = mealData[[0, 5, 10, 15, 20, 25]]
    meal_window3 = mealData[[30, 35, 40, 45, 50, 55]]
    meal_window4 = mealData[[60, 65, 70, 75, 80, 85]]
    meal_window5 = mealData[[90, 95, 100, 105, 110, 115]]

    for i in range(len(mealData)):

        mealMaxIndex = mealData.iloc[i].loc[mealData.iloc[i] == mealData.iloc[i].max()].index.tolist()[0]
        mealMinIndex = mealData.iloc[i].loc[mealData.iloc[i] == mealData.iloc[i].min()].index.tolist()[0]
        tmax_tmin.append(mealMaxIndex - mealMinIndex)

        meal_diff_cgm = mealData.iloc[i].max() - mealData.iloc[i].min()
        max_min.append(meal_diff_cgm)

        derivative.append((mealData.iloc[i].diff() / 5).max())

        rolling_mean.append(mealData.iloc[i].rolling(window=5).mean().mean())

        window_mean.append((meal_window1.iloc[i].mean() + meal_window2.iloc[i].mean() + meal_window3.iloc[i].mean() +
                            meal_window4.iloc[i].mean() + meal_window5.iloc[i].mean()) / 5)

        labels.append(1)
    nomeal_window1 = noMealData[[0, 5, 10, 15, 20, 25]]
    nomeal_window2 = noMealData[[30, 35, 40, 45, 50, 55]]
    nomeal_window3 = noMealData[[60, 65, 70, 75, 80, 85]]
    nomeal_window4 = noMealData[[90, 95, 100, 105, 110, 115]]

    for i in range(len(noMealData)):
        noMealMaxIndex = noMealData.iloc[i].loc[noMealData.iloc[i] == noMealData.iloc[i].max()].index.tolist()[0]
        noMealMinIndex = noMealData.iloc[i].loc[noMealData.iloc[i] == noMealData.iloc[i].min()].index.tolist()[0]
        tmax_tmin.append(noMealMaxIndex - noMealMinIndex)

        nomeal_diff_cgm = noMealData.iloc[i].max() - noMealData.iloc[i].min()
        max_min.append(nomeal_diff_cgm)

        derivative.append((noMealData.iloc[i].diff() / 5).max())

        rolling_mean.append(noMealData.iloc[i].rolling(window=5).mean().mean())

        window_mean.append(
            (nomeal_window1.iloc[i].mean() + nomeal_window2.iloc[i].mean() + nomeal_window3.iloc[i].mean() +
             nomeal_window4.iloc[i].mean()) / 5)

        labels.append(0)

    featureMatrix = pd.DataFrame(columns=[1, 2, 3, 4, 5, 'class'], index=range(len(mealData) + len(noMealData)))
    featureMatrix[1] = tmax_tmin
    featureMatrix[2] = max_min
    featureMatrix[3] = derivative
    featureMatrix[4] = rolling_mean
    featureMatrix[5] = window_mean
    featureMatrix['class'] = labels

    X_train, X_test, y_train, y_test = train_test_split(featureMatrix[[1, 2, 3, 4, 5]], featureMatrix[['class']],
                                                        test_size=0.33, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = svm.SVC()
    clf.fit(X_train, y_train)
    pred_clf = clf.predict(X_test)

    pickle_out = open('train.pickle', 'wb')
    pickle.dump(clf, pickle_out)
    pickle_out.close()

    print(classification_report(y_test, pred_clf))
    print(confusion_matrix(y_test, pred_clf))


if __name__ == '__main__':
    run_script()
