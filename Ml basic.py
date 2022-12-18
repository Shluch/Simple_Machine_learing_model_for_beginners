import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #for prediction
from sklearn.metrics import mean_squared_error,r2_score #for evaluation
from sklearn.ensemble import RandomForestRegressor  #for random forest

#loading data
df= pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
print(df)

#spliting x and y axis 

y = df['logS']
print(y)

x = df.drop('logS',axis=1) #removing logS coloum from table as we stored it in y already axis= 1 means coloum
print(x) #new dataset 

#trainig and testing 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100) #data spliting
print(x_train) #it have 80% of data now 
print(x_test) # it have 20% of data 

#NOW LETS BUILD MODEL 

lr = LinearRegression()
lr.fit(x_train,y_train)

#Apply prediction
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

print(y_lr_train_pred,y_lr_test_pred)

#Evaluting model performance

lr_train_mse = mean_squared_error(y_train,y_lr_train_pred)
lr_train_r2 = r2_score(y_train,y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test,y_lr_test_pred)
lr_test_r2 = r2_score(y_test,y_lr_test_pred)

print(lr_train_mse,lr_train_r2)
print(lr_test_mse,lr_test_r2)

#Random Forest                                       #Random forest use data find the majority of set to give out come

rf = RandomForestRegressor(max_depth=2,random_state=100)
rf.fit(x_train,y_train)

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

print( y_rf_test_pred,y_rf_train_pred)

rf_train_mse = mean_squared_error(y_train,y_rf_train_pred)
rf_train_r2 = r2_score(y_train,y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test,y_rf_test_pred)
rf_test_r2 = r2_score(y_test,y_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest: ',rf_train_mse,rf_train_r2,rf_test_r2,rf_test_mse]).transpose()
rf_results.columns = ['Methods','Training MSe','Traing r2','test mse','test r2']
print(rf_results)

#we used two diffrent models random foresrt and linear regression both gave us similar kind of output
 

#DATA REPRESNTATION
plt.scatter(x=y_train,y=y_lr_train_pred,alpha = .3)  #Assiging x and y lines value alpha is opacity 

print(plt.plot())