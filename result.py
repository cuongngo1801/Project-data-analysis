import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import math
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import Eval 

def create_label():
	df = pd.read_csv('InsulinData.csv', low_memory=False, 
					usecols=['Index', 'Date', 'Time', 'BWZ Carb Input (grams)'])
	cgm = pd.read_csv('CGMData.csv', low_memory=False,
						usecols=['Index', 'Date', 'Time', 'Sensor Glucose (mg/dL)'])
	y_columns = df['BWZ Carb Input (grams)']
	y_columns = y_columns.loc[(y_columns!=0.0)]
	bin_size = 20.0
	#changing data types
	cgm['DateTime'] = cgm['Date'] + ' ' + cgm['Time']

	cgm['DateTime'] = pd.to_datetime(cgm['DateTime'], format="%m/%d/%Y %H:%M:%S")

	cgm['Time'] = pd.to_datetime(cgm['Time'], format="%H:%M:%S").dt.time
   
	df['DateTime'] = df['Date'] + ' ' + df['Time']
	df['DateTime'] = pd.to_datetime(df['DateTime'], format="%m/%d/%Y %H:%M:%S")
	df['Time'] = pd.to_datetime(df['Time'], format="%H:%M:%S").dt.time

	bin_label = pd.Series(index=range(len(y_columns)))
	max = float(y_columns.max())
	min = float(y_columns.min())
	bin_startpoint = min
	NumberOfBin = math.floor((max-min)/bin_size)
	for i in range(NumberOfBin):
		if i != NumberOfBin:
			bin_index = list(y_columns[(y_columns >= bin_startpoint) & (y_columns < bin_startpoint + bin_size)].index)
		else: 
			bin_index = list(y_columns[(y_columns >= bin_startpoint) & (y_columns < max)].index)
		for j in range(len(bin_index)):
			bin_label[bin_index[j]]  = i
		bin_startpoint = bin_startpoint + bin_size
	
	df['Bin_label'] = bin_label
	df.dropna(inplace=True)
	df = \
        df.loc[(df['BWZ Carb Input (grams)'].isna() == False) & (df['BWZ Carb Input (grams)'] != 0)][
            ['DateTime','Bin_label']]
	df['DiffInMeals'] = df['DateTime'].diff(-1).to_frame()
	mealStartTime = df.loc[df['DiffInMeals'] > '02:00:00']

	#Save data bin label
	bin_label = pd.DataFrame(data= mealStartTime)
	bin_label.to_csv('binlabel.csv', header = None, index = True)
	# print(mealStartTime)


	# Cluster
	Feature_root, bins = get_extraction(mealStartTime,cgm) 
	return Feature_root, bins
def get_extraction(mealStartTime,cgm):
    # extract the meal time from cgm and get the glucose reading
    mealData1Cols = list(range(-30, 120, 5))

    mealData = pd.DataFrame(columns=mealData1Cols, index=range(len(mealStartTime)))
    for i in range(len(mealStartTime) - 2):
        mealStartTemp = cgm.loc[cgm['DateTime'] >= mealStartTime.iloc[i]['DateTime']]
        mealStartIndex = len(mealStartTemp) - 1
        preMealStartIndex = mealStartIndex + 6
        postMealStartIndex = mealStartIndex - 23
        meal_i = cgm.iloc[postMealStartIndex:preMealStartIndex + 1, :]['Sensor Glucose (mg/dL)'].iloc[::-1].to_numpy()
        mealData.iloc[i] = meal_i 

    mealStartTime1 = mealStartTime['Bin_label']
    meal_bin = mealStartTime1.reset_index()
    meal_bin.drop('index', axis = 1, inplace = True)
    # print(xyt)


    new_meal = pd.concat([mealData,meal_bin] ,axis=1, join_axes=[mealData.index])
    new_meal.dropna(inplace=True)
    new_meal = pd.DataFrame(data= new_meal)
    new_meal.to_csv('new_meal.csv', header = None, index = True)
    # print(new_meal)
    meal_bins = pd.DataFrame(index = new_meal)
    meal_bins = new_meal['Bin_label']
    meal_bins = meal_bins.reset_index()
    meal_bins.drop('index', axis = 1, inplace = True)
    mealData.dropna(inplace=True)	
    # print(meal_bins)
    # MD = pd.DataFrame(data= mealData)
    # MD.to_csv('MD.csv', header = None, index = None)
    Feature,bin_featureMatrix = get_feature(mealData,meal_bins)
    return Feature , bin_featureMatrix

def get_feature(mealData,meal_bins):
    tmax_tmin = []
    max_min = []
    derivative = []
    rolling_mean = []
    window_mean = []
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

    featureMatrix = pd.DataFrame(columns=[1, 2, 3, 4, 5], index=range(len(mealData)))
    featureMatrix[1] = tmax_tmin
    featureMatrix[2] = max_min
    featureMatrix[3] = derivative
    featureMatrix[4] = rolling_mean
    featureMatrix[5] = window_mean
    # FM = pd.DataFrame(data= featureMatrix)
    # FM.to_csv('FM.csv', header = None, index = None)
    bin_featureMatrix = pd.DataFrame(index = featureMatrix)
    bin_featureMatrix = pd.concat([featureMatrix,meal_bins], axis = 1, join_axes= [meal_bins.index])
    # print(bin_featureMatrix)
    return featureMatrix, bin_featureMatrix

def Kmeans():
	X,bins = create_label()

	km = KMeans(n_clusters=6,        
            init='k-means++',        
            n_init=10,               
            max_iter=300,            
            tol=1e-04,        
            random_state=0) 

	y_km = km.fit_predict(X)
	print(km.inertia_)
	# print(y_km, '\n', len(y_km))
	bins['kmean'] = y_km
	# bins.to_csv('bins.csv', header = None, index = True)
	# print(bins)
	return bins
def DB_SCAN():
    X,bins = create_label()
    db = DBSCAN(eps= 35, min_samples=10)
    db.fit_predict(X) 
    labels = db.labels_
    n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
    db = db.fit_predict(X)
    bins['DBSCAN'] = db
    print(n_cluster)
    return bins , n_cluster , db
def DB_SCAN_Bisect():
    bins , n_cluster , db = DB_SCAN()
    while ( n_cluster< 6 ):
        matrixes_index = []
        matrixes = []
        SSEs = []
        for i in range(n_cluster):
            matrixes_index.append(bins.loc[bins['DBSCAN'] == i].index)
            matrix = bins.loc[bins['DBSCAN'] == i]
            matrix = matrix.drop(['Bin_label','DBSCAN'],axis = 1)
            matrix = matrix.to_numpy()
            SSEs.append(Eval.SSE(matrix))
            matrixes.append(matrix)
        SSEs = np.array(SSEs)
        max_cluster = np.where(SSEs == np.max(SSEs))
        km = KMeans(n_clusters=2,        # số cluster
                    init='k-means++',           # vị trí center của cluster  default: 'k-means++'
                    n_init=10,               # số lần chọn center của cluster default: '10'  trong số lần chọn  , sẽ chọn ra model có  SSE nhỏ nhất
                    max_iter=300,            # Tiến hành chạy k-means nhiều nhất bao nhiêu lần default: '300'
                    tol=1e-04,               # Khi tiến hành hội tụ các điểm, sai số cho phép là bao nhiêu, default: '1e-04'
                    random_state=0)
        bisect = km.fit_predict(matrixes[max_cluster[0][0]])
        for i in range(len(matrixes_index[max_cluster[0][0]])):
            if bisect[i] == 0:
                a = matrixes_index[max_cluster[0][0]][i]
                bins.loc[a,'DBSCAN'] = max_cluster[0][0]
            elif bisect[i] == 1:
                a = matrixes_index[max_cluster[0][0]][i]
                bins.loc[a,'DBSCAN'] = n_cluster
        n_cluster += 1
    return bins
def SSE(feature_matrix):
    mean_vector = feature_matrix.mean(axis = 0)
    squared_mean_vector = np.square(mean_vector)
    squared_mean = np.sum(squared_mean_vector)
    SSE = 0.0
    for i in range(np.shape(feature_matrix)[0]):
        value1_vector = np.square(feature_matrix[i][:])
        value1 = np.sum(value1_vector)
        SSE = SSE + (value1 - squared_mean)
    return SSE
#def create_eval_matrix(cluster, bin ):

def find_entropy(matrix):
    total_entropy = 0
    sums = []
    entropy = []
    total_sum = np.sum(matrix) 
    for i in range(6):
        minus_entropy = 0
        sums.append(np.sum(matrix[i][:]))
        for j in range(6):
            a = matrix[i][j]/sums[i]
            minus_entropy = minus_entropy + a * math.log(a)
            local_entropy = - minus_entropy
        entropy.append(local_entropy)
        total_entropy = total_entropy + (sums[i]/total_sum) * entropy[i]
    return total_entropy

def find_purity(matrix):
    n = np.sum(matrix)
    maxs = 0 
    for i in range(6):
        maxs = maxs + np.max(matrix[i][:])
    purity = (1/n) * maxs
    return purity

# print(find_purity(matrix))       
if __name__ == '__main__':
	# create dataFrame for caculator SSE()
	# print(Matrix_SCAN)
	matrix_feature = Kmeans()
	matrix_window0 = pd.DataFrame(index=range(len(matrix_feature)))
	matrix_window1 = pd.DataFrame(index=range(len(matrix_feature)))
	matrix_window2 = pd.DataFrame(index=range(len(matrix_feature)))
	matrix_window3 = pd.DataFrame(index=range(len(matrix_feature)))
	matrix_window4 = pd.DataFrame(index=range(len(matrix_feature)))
	matrix_window5 = pd.DataFrame(index=range(len(matrix_feature)))

	matrix_window0 = matrix_feature.loc[(matrix_feature['kmean'] == 0)]
	matrix_window1 = matrix_feature.loc[(matrix_feature['kmean'] == 1)]
	matrix_window2 = matrix_feature.loc[(matrix_feature['kmean'] == 2)]
	matrix_window3 = matrix_feature.loc[(matrix_feature['kmean'] == 3)]
	matrix_window4 = matrix_feature.loc[(matrix_feature['kmean'] == 4)]
	matrix_window5 = matrix_feature.loc[(matrix_feature['kmean'] == 5)]
	
    #Take feature of DB_SCAN
	Matrix_SCAN = DB_SCAN_Bisect()

	SCAN_window0 = pd.DataFrame(index=range(len(Matrix_SCAN)))
	SCAN_window1 = pd.DataFrame(index=range(len(Matrix_SCAN)))
	SCAN_window2 = pd.DataFrame(index=range(len(Matrix_SCAN)))
	SCAN_window3 = pd.DataFrame(index=range(len(Matrix_SCAN)))
	SCAN_window4 = pd.DataFrame(index=range(len(Matrix_SCAN)))
	SCAN_window5 = pd.DataFrame(index=range(len(Matrix_SCAN)))


	SCAN_window0 = Matrix_SCAN.loc[(Matrix_SCAN['DBSCAN'] == 0)]
	SCAN_window1 = Matrix_SCAN.loc[(Matrix_SCAN['DBSCAN'] == 1)]
	SCAN_window2 = Matrix_SCAN.loc[(Matrix_SCAN['DBSCAN'] == 2)]
	SCAN_window3 = Matrix_SCAN.loc[(Matrix_SCAN['DBSCAN'] == 3)]
	SCAN_window4 = Matrix_SCAN.loc[(Matrix_SCAN['DBSCAN'] == 4)]
	SCAN_window5 = Matrix_SCAN.loc[(Matrix_SCAN['DBSCAN'] == 5)]

	#Drop matrix kmean cluster
	matrix_window0.drop(['kmean','Bin_label'], axis = 1, inplace = True)
	matrix_window1.drop(['kmean','Bin_label'], axis = 1, inplace = True)
	matrix_window2.drop(['kmean','Bin_label'], axis = 1, inplace = True)
	matrix_window3.drop(['kmean','Bin_label'], axis = 1, inplace = True)
	matrix_window4.drop(['kmean','Bin_label'], axis = 1, inplace = True)
	matrix_window5.drop(['kmean','Bin_label'], axis = 1, inplace = True)
	

	#Drop matrix DBSCAN cluster
	SCAN_window0.drop(['DBSCAN','Bin_label'], axis = 1, inplace = True)
	SCAN_window1.drop(['DBSCAN','Bin_label'], axis = 1, inplace = True)
	SCAN_window2.drop(['DBSCAN','Bin_label'], axis = 1, inplace = True)
	SCAN_window3.drop(['DBSCAN','Bin_label'], axis = 1, inplace = True)
	SCAN_window4.drop(['DBSCAN','Bin_label'], axis = 1, inplace = True)
	SCAN_window5.drop(['DBSCAN','Bin_label'], axis = 1, inplace = True)

	#Caculator SSE of Kmean
	SSE0 = SSE(matrix_window0.to_numpy())
	SSE1 = SSE(matrix_window1.to_numpy())
	SSE2 = SSE(matrix_window2.to_numpy())
	SSE3 = SSE(matrix_window3.to_numpy())
	SSE4 = SSE(matrix_window4.to_numpy())
	SSE5 = SSE(matrix_window5.to_numpy())
	
	SSE = SSE0 + SSE1 + SSE2 + SSE3 + SSE4 + SSE5
	#Caculator SSE of SCAN
	SSE0_SCAN = SSE(SCAN_window0.to_numpy())
	SSE1_SCAN = SSE(SCAN_window1.to_numpy())
	SSE2_SCAN = SSE(SCAN_window2.to_numpy())
	SSE3_SCAN = SSE(SCAN_window3.to_numpy())
	SSE4_SCAN = SSE(SCAN_window4.to_numpy())
	SSE5_SCAN = SSE(SCAN_window5.to_numpy())
	
	SSE_SCAN = SSE0_SCAN +SSE1_SCAN + SSE2_SCAN + SSE3_SCAN + SSE4_SCAN
	print(SSE_SCAN)
	# print(SSE)


	