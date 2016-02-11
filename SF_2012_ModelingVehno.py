import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import glob, os
import pylab
import timeit
import sklearn
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import chi2

#apc_1023
start = timeit.default_timer()
store = pd.HDFStore(r'R:\Angela\fast_trips\apc_1203.h5')
data1 = store['APCdata']
stop = timeit.default_timer()
print 'apc_1203:', stop - start #apc_1203: 63.1637678053
#apc_1206
start = timeit.default_timer()
store = pd.HDFStore(r'R:\Angela\fast_trips\apc_1206.h5')
data2 = store['APCdata']
stop = timeit.default_timer()
print 'apc_1206:', stop - start #apc_1206: 72.5603196086
#apc_1209
start = timeit.default_timer()
store = pd.HDFStore(r'R:\Angela\fast_trips\apc_1209.h5')
data3 = store['APCdata']
stop = timeit.default_timer()
print 'apc_1209:', stop - start #apc_1209: 99.1659132181

print 'finish reading data'


#keep the useful columns 
def data_need(data):
    data = data[['on', 'off', 'vehno', 'route', 'stop_name', 'stop_sequence', 'mo', 'yr', 'arr_hour', 'arr_min', 'arr_sec', 'dep_hr', 'dep_min', 'dep_sec']]
    return data

data1 = data_need(data1)
data2 = data_need(data2)
data3 = data_need(data3)

#combine datasets (to speed up data reading time in the future ?)
#combined_data = pd.concat([data1,data2, data3])
#combined_data.to_csv(r'R:\Angela\fast_trips\apc_1203_1206_1209.csv')
#read the combined data if you want to 
data = pd.read_csv(r'R:\Angela\fast_trips\apc_1203_1206_1209.csv')


#prepare bus info
vehicles = pd.read_csv(r'R:\Angela\fast_trips\Vehicles.csv')
fleet = pd.read_csv(r'R:\Angela\fast_trips\Copy of Fleet.csv')
vehicles.Artic = vehicles.Length.map({60 : 1, 40 : 0, 30 : 0})
vehicles.Floor = vehicles['Low Floor'].map({'Y': 1, 'N' : 0})
vehicles.loc[:,'Artic'] = pd.Series(vehicles.Artic, index=vehicles.index)
vehicles.loc[:,'Floor'] = pd.Series(vehicles.Floor, index=vehicles.index)
df_artic = vehicles.set_index('Equip_Type').to_dict()['Artic']
df_floor = vehicles.set_index('Equip_Type').to_dict()['Floor']
df_doors = vehicles.set_index('Equip_Type').to_dict()['Doors']
fleet['Doors'] = fleet['Equip_Type'].map(df_doors)
fleet['Artic'] = fleet['Equip_Type'].map(df_artic)
fleet['Floor'] = fleet['Equip_Type'].map(df_floor)
df_vehnum_doors = fleet.set_index('VehNum').to_dict()['Doors']
df_vehnum_artic = fleet.set_index('VehNum').to_dict()['Artic']
df_vehnum_floor = fleet.set_index('VehNum').to_dict()['Floor']
#prepare route type
route_type = pd.read_csv(r'R:\Angela\fast_trips\MuniRouteTypes.csv')
route_type = route_type.dropna()
dict_route_type = {}
dict_route_type = route_type.set_index('APC Route ID')['Type'].to_dict()

#prepare variables
def get_x_y(data):
    start = timeit.default_timer()
    #Get rid of rows where certain fields has null/nan values
    data = data.dropna(subset = ['on', 'off', 'vehno'])
    #input bus type info into raw data
    #COMPUTE EOL = RINDEX(ANAME,' - EOL') 
    data['EOL']= data.apply(lambda x: '- EOL' in x['stop_name'], axis=1).map({False: 1, True: 0})
    #remove the last stop
    data = data.loc[data['EOL'] == 1]
    #remove the first stop
    data = data.loc[data['stop_sequence'] != 1]
    stop = timeit.default_timer()
    print 'clean data:', stop - start

    #COMPUTE TIMESTOP=((HR * 3600) + (MIN * 60) + SEC)
    start = timeit.default_timer()
    data['COMPUTE_TIMESTOP'] = data['arr_hour']*3600 + data['arr_min']*60 + data['arr_sec']
    #COMPUTE DOORCLOSE=(( DHR * 3600) + (DMIN * 60) + DSEC)
    data['COMPUTE_DOORCOLSE'] = data['dep_hr']*3600 + data['dep_min']*60 + data['dep_sec']
    #COMPUTE DOORDWELL=DOORCLOSE-TIMESTOP
    data['COMPUTE_DOORDWELL'] = data['COMPUTE_DOORCOLSE'] - data['COMPUTE_TIMESTOP']
    #appling door dwell time less than 120 secs
    data = data.loc[data['COMPUTE_DOORDWELL'] <= 90]
    stop = timeit.default_timer()
    print 'compute dwell time:', stop - start

    #Keep rows that satisfy a query:
    start = timeit.default_timer()
    data['Doors'] = data['vehno'].map(df_vehnum_doors) 
    data['Artic'] = data['vehno'].map(df_vehnum_artic)
    data['Floor'] = data['vehno'].map(df_vehnum_floor)
    data['two_doors'] = data['Doors'].map({2: 1, 3: 0})
    data['three_doors'] = data['Doors'].map({2: 0, 3: 1})
    data['all_door_boarding']= data.apply(lambda x: x['mo'] > 6, axis=1).map({False: 0, True: 1})
    #make dummie variables for route id
    data['Route Type'] = data['route'].map(dict_route_type)
    just_dummies_route = pd.get_dummies(data['Route Type'])
    step_1 = pd.concat([data, just_dummies_route], axis=1)
    step_1.drop(['Local'], inplace=True, axis=1)
    data = step_1
    stop = timeit.default_timer()
    print 'add veh&route info:', stop - start

    #Create dummie variables for bus id 
    start = timeit.default_timer()
    just_dummies_veh = pd.get_dummies(data['vehno'])
    step_1 = pd.concat([data, just_dummies_veh], axis=1)
    #get rid of one dummy variable to avoid the dummy variable trap
    step_1.drop([8515], inplace=True, axis=1)
    data = step_1
    stop = timeit.default_timer()
    print 'add bus id variables:', stop - start

    #Create interaction variables
    start = timeit.default_timer()
    data['on_threedoors'] = data['on']*data['three_doors']
    data['off_threedoors'] = data['off']*data['three_doors']
    data['on_floor'] = data['on']*data['Floor']
    data['off_floor'] = data['off']*data['Floor']
    data['floor_threedoors'] = data['Floor']*data['three_doors']
    data['floor_twodoors'] = data['Floor']*data['two_doors']
    data['on_all_door_boarding'] = data['on']*data['all_door_boarding']
    data['off_all_door_boarding'] = data['off']*data['all_door_boarding']
    data['on_express'] = data['on']*data['Express']
    data['off_express'] = data['off']*data['Express']
    data['on_rapid'] = data['on']*data['Rapid']
    data['off_rapid'] = data['off']*data['Rapid']
    data['on_owl'] = data['on']*data['OWL']
    data['off_owl'] = data['off']*data['OWL']
    stop = timeit.default_timer()
    print 'add interaction variables:', stop - start

    return data



############### Ramdomly pick up (test_percentage)% data to model ############
#note: applied: combined datasets, which are apc_1203 data and apc_1206 data
#test 10% of the data
test_percent = 0.1 
test_percent_title = str(test_percent)
msk = np.random.rand(len(combined_data)) < (1-test_percent)
test = combined_data[~msk]
print len(test)

data = get_x_y(test)
print 'finished data preparation'


################################# build model #################################

#####model with vehicle ids 
#Run Linear Regression In Python SciKit-Learn
start = timeit.default_timer()
X = data.drop(['COMPUTE_DOORDWELL','vehno','stop_name', 'stop_sequence', 'arr_hour','arr_min','arr_sec','route','mo','yr', 'dep_hr','dep_min','dep_sec', 'EOL','COMPUTE_TIMESTOP','COMPUTE_DOORCOLSE','COMPUTE_DOORDWELL','Doors','two_doors', 'Artic', 'Route Type', 'floor_threedoors', 'floor_twodoors'], axis=1)
# impute missing values
X[np.isnan(X)] = np.median(X[~np.isnan(X)])
y = data.COMPUTE_DOORDWELL

#training and Validating dataset
#train 90% data
#validate 10% data
length = len(X) * 0.1
length_index = (-1) * int(length)
X_train = X[:length_index]
X_test = X[length_index:]
y_train = y[:length_index]
y_test = y[length_index:]
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape

#create a LinearRegression object 
lm1 = LinearRegression()
lm1.fit(X_train, y_train)
pred_train = lm1.predict(X_train)
pred_test = lm1.predict(X_test)
LinearRegression(copy_X=True, fit_intercept=True,normalize=False)

#construct a data frame that contains features and estimated coefficients
sklearn_results = pd.DataFrame(zip(X_train.columns, lm1.coef_), columns = ['features', 'estimatedCoefficients'])
#p-values
scores, pvalues = chi2(X_train, y_train)
sklearn_results['pvalues'] = pd.DataFrame(pvalues)
sklearn_results['scores'] = pd.DataFrame(scores)
sklearn_results.to_csv('R:/Angela/fast_trips/muni_apc_2012/10%_del_8515_of_1203_1206_datasets_datasets_withVeh.csv')
stop = timeit.default_timer()
print 'build model with veh ids:', stop - start

#residual plots
plt.scatter(lm1.predict(X_train), lm1.predict(X_train) - y_train, c='y', s=20)
plt.scatter(lm1.predict(X_test), lm1.predict(X_test) - y_test, c='r', s=20)
#plt.hlines(y=0, xmin=0, xmax=50)
plt.axis([-50, 300, -120, 120])
plt.title(test_percent_title + 'data: Residual Plot using 90% training (yellow) and 10% test (red) data')
plt.ylabel('Residuals')
pylab.savefig(r'R:\Angela\fast_trips\muni_apc_2012\10%_del_8515_of_1203_1206_datasets_withVeh.png')



######model without the vehicle IDs
start = timeit.default_timer()
X = data[['on','off','Floor','three_doors','all_door_boarding','Express','OWL','Rapid','on_threedoors','off_threedoors','on_floor','off_floor','on_all_door_boarding','off_all_door_boarding','on_express','off_express','on_rapid','off_rapid','on_owl','off_owl']]
X[np.isnan(X)] = np.median(X[~np.isnan(X)])
y = data.COMPUTE_DOORDWELL

#training and Validating dataset
X_train = X[:length_index]
X_test = X[length_index:]
y_train = y[:length_index]
y_test = y[length_index:]
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape

#creates a LinearRegression object 
lm2 = LinearRegression()
lm2.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True,normalize=False)

#construct a data frame that contains features and estimated coefficients
sklearn_results = pd.DataFrame(zip(X_train.columns, lm2.coef_), columns = ['features', 'estimatedCoefficients'])
#p-values
scores, pvalues = chi2(X_train, y_train)
sklearn_results['pvalues'] = pd.DataFrame(pvalues)
sklearn_results['scores'] = pd.DataFrame(scores)
sklearn_results.to_csv('R:/Angela/fast_trips/muni_apc_2012/10%_del_8515_of_1203_1206_datasets_datasets_withoutVeh10.csv')
stop = timeit.default_timer()
print 'build model without veh ids:', stop - start

#residual plots
plt.scatter(lm2.predict(X_train), lm2.predict(X_train) - y_train, c='y', s=20)
plt.scatter(lm2.predict(X_test), lm2.predict(X_test) - y_test, c='r', s=20)
#plt.hlines(y=0, xmin=0, xmax=50)
#plt.axis([-50, 300, -120, 120])
plt.title(test_percent_title + 'data: Residual Plot using 90% training (yellow) and 10% test (red) data')
plt.ylabel('Residuals')
pylab.savefig(r'R:\Angela\fast_trips\muni_apc_2012\10%_del_8515_of_1203_1206_datasets_datasets_withoutVeh10.png')

print 'end'