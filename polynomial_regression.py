import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#define objects
poly_feautres=PolynomialFeatures(degree=3)
lm=LinearRegression()

#set seed
np.random.seed(42)

#generate random data
x=np.sort(5*np.random.rand(80,1),axis=0)
y=np.sin(x)+np.random.normal(0,0.1,x.shape)


#split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#create polynomial features
x_poly_train=poly_feautres.fit_transform(x_train)
x_poly_test=poly_feautres.fit_transform(x_test)

lm.fit(x_train,y_train)
predictions=lm.predict(x_test)
print(mean_absolute_error(y_test,predictions))

lm.fit(x_poly_train,y_train)
predictions=lm.predict(x_poly_test)
print(mean_absolute_error(y_test,predictions))