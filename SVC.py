
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
  

data_set= pd.read_csv('suv_data.csv')  

x= data_set.iloc[:, [2,3]].values  
y= data_set.iloc[:, 4].values  
  
  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  

from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)   


""" from sklearn.svm import SVC # "Support vector classifier"  
classifier = SVC(kernel='linear', random_state=0)  
classifier.fit(x_train, y_train)  """ 
from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(x_train, y_train)  

y_pred= classifier.predict(x_test)  

print(y_pred)
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)  

print(cm) 
"""
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1:].values

dataset.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

print(dataset)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
regressor.score(X_test,y_test)



import matplotlib.pyplot as plt

plt.scatter(X_train, y_train, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show() """