# -*- coding: utf-8 -*-
"""
Prepare Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import math
#imports for SVM (Support Vector Machines)
from seaborn import load_dataset, pairplot
from seaborn import scatterplot
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


"""load and prepare data"""
#read raw data into pandas Dataframe 
df_base = pd.read_excel("C:\\Users\\anna-\\Nextcloud2\\DHBW\\2. Sem\\DataScienceProject\\dataexport_20220730T081818.xlsx", engine='openpyxl')
#drop first 9 unneeded columns
df_base = df_base.drop(labels = np.arange(0,9), axis = 0)
#reset index (starting from 0)
df_base = df_base.reset_index(drop=True)
#rename columns
df_base.set_axis(["day","precipitation","temp_min","temp_max"], axis=1, inplace=True)
#format timestamp in raw data and insert new columns for month and day 
df_base.insert(0, "month", df_base["day"].astype('string').str.replace('00:00:00','').str.split('-').str[1].astype('int'))
df_base.insert(0, "year", df_base["day"].astype('string').str.replace('00:00:00','').str.split('-').str[0].astype('int'))
df_base["day"] = df_base["day"].astype('string').str.replace('00:00:00','').str.split('-').str[2].astype('int')
df_base = df_base.dropna(axis=0)


# In[127]:

years = np.arange(1984,2023)
months = np.arange(1, 13)

# create DataFrames for average data
df_av_yearly = {"index":[], "year":[],"temp_min":[], "temp_max":[], "precipitation":[]}
df_av_monthly = {"index":[],"y-m": [], "month":[], "temp_min":[], "temp_max":[], "precipitation":[]}
index_years = 1
index_months = 1

for y in years:
    df_av_yearly["index"].append(index_years)
    df_av_yearly["year"].append(y)
    df_av_yearly["temp_min"].append((df_base["temp_min"][df_base["year"] == y]).mean())
    df_av_yearly["temp_max"].append((df_base["temp_max"][df_base["year"] == y]).mean())
    df_av_yearly["precipitation"].append((df_base["precipitation"][df_base["year"] == y]).mean())
    index_years += 1
    
    for m in months:
        df_av_monthly["index"].append(index_months)
        df_av_monthly["y-m"].append(f"{y}-{m}")
        df_av_monthly["month"].append(m)
        df_av_monthly["temp_min"].append((df_base["temp_min"][(df_base["year"] == y) & (df_base["month"] == m)]).mean())
        df_av_monthly["temp_max"].append((df_base["temp_max"][(df_base["year"] == y) & (df_base["month"] == m)]).mean())
        df_av_monthly["precipitation"].append((df_base["precipitation"][(df_base["year"] == y) & (df_base["month"] == m)]).mean())
        index_months += 1
        
df_av_yearly = pd.DataFrame.from_dict(df_av_yearly).dropna(axis=0)
df_av_monthly = pd.DataFrame.from_dict(df_av_monthly).dropna(axis=0)




""" -------------------------------------------------------------------------------"""
"""linear Regression """
""" -------------------------------------------------------------------------------"""

# In[188]:


def linear_regression(x, y): 
    # get number of observations in dataset
    N = len(x)
    
    # calculate means
    x_mean = x.mean()
    print(x_mean)
    y_mean = y.mean()
    print(y_mean)
    
    # calculate theta1: calculate numerator / denominator
     
    T1_num = ((x - x_mean) * (y - y_mean)).sum()
    T1_den = ((x - x_mean)**2).sum()
    T1 = T1_num / T1_den
    
    # calcualte theta0
    T0 = y_mean - (T1*x_mean)
    # regression line rounded to 20 decimal places    
    reg_line = 'y = {} + {}x'.format(T0, round(T1, 20))
    
    return (T0, T1, reg_line)



#data = pd.DataFrame({'x': [0,1,2,3,4,5,6], 'y': [1,2,3,4,5,6,7]})
y = df_av_monthly["temp_max"][df_av_monthly["month"] == 8].values
print(y)

x = pd.DataFrame(np.arange(0,len(y))).values
#print(x)
T0, T1, reg_line = linear_regression(x,y)
print(reg_line)



# In[189]:


# calculating how well the line fits: 
# correlation coefficient R and coefficient of determination R^2

# Pearson's correlation coefficient
def corr_coef(x, y):
    N = len(x)
    
    num = (N * (x*y).sum()) - (x.sum() * y.sum())
    den = np.sqrt((N * (x**2).sum() - x.sum()**2) * (N * (y**2).sum() - y.sum()**2))
    R = num / den
    return R


print('Regression Line: ', reg_line)
R = corr_coef(x, y)
print('Correlation Coef.: ', R)
print('"Goodness of Fit": ', R**2)





# SIMPLE PLOT
plt.figure()
plt.scatter(x,y)
plt.plot(x, T0 + T1*x,color = 'red')
plt.title('Linear Regression from scratch')
plt.xlabel('maximal Temperature in August between 1984 ans 2021')
plt.ylabel('maximal Temperature in degrees celcius')
plt.show()

# In[ ]:


# make predictions
def predict(new_x, T0=T0, T1=T1):
    y = T0 + T1 * new_x
    return y


""" -------------------------------------------------------------------------------"""
""" Linear Regression with sklearn"""
""" -------------------------------------------------------------------------------"""
# In[191]:


""" Using sklearn """

lin = LinearRegression()
lin.fit(x, y)
# Visualising the Linear Regression results
plt.scatter(x, y, color = 'blue')
 
plt.plot(x, lin.predict(x), color = 'red')
plt.title('Linear Regression using sklearn')
plt.xlabel('maximal Temperature in August between 1984 ans 2021')
plt.ylabel('maximal Temperature in degrees celcius')
 
plt.show()


 
poly = PolynomialFeatures(degree = 15)
X_poly = poly.fit_transform(x)
 
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)

# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'blue')
 
plt.plot(x, lin2.predict(poly.fit_transform(x)), color = 'red')
plt.title('Polynomial Regression using sklearn')
plt.xlabel('years starting from 1984')
plt.ylabel('maximal monthly Temperature (August)')
 
plt.show()





""" -------------------------------------------------------------------------------"""
""" Polynomial Regression: simple approach"""
""" -------------------------------------------------------------------------------"""
# In[190]:


y2 = df_av_yearly["temp_max"].values
mymodel = np.poly1d(np.polyfit(np.array(np.arange(0,len(y))), y, len(y)*2))

myline = np.linspace(1, 30, 150)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline), color = 'red')
plt.title('Polynomial Regression using np.poly1d')
plt.xlabel('years starting from 1984')
plt.ylabel('maximal monthly Temperature (August)')
plt.show() 


""" -------------------------------------------------------------------------------"""
""" Polynomial Regression from scratch"""
""" -------------------------------------------------------------------------------"""
# In[192]:
#https://www.geeksforgeeks.org/polynomial-regression-from-scratch-using-python/

class PolynomailRegression() :
     
    def __init__( self, degree, learning_rate, iterations ) :
         
        self.degree = degree
         
        self.learning_rate = learning_rate
         
        self.iterations = iterations
         
    # function to transform X
     
    def transform( self, X ) :
         
        # initialize X_transform
         
        X_transform = np.ones( ( self.m, 1 ) )
         
        j = 0
     
        for j in range( self.degree + 1 ) :
             
            if j != 0 :
                 
                x_pow = np.power( X, j )
                 
                # append x_pow to X_transform
                 
                X_transform = np.append( X_transform, x_pow.reshape( -1, 1 ), axis = 1 )
 
        return X_transform  
     
    # function to normalize X_transform
     
    def normalize( self, X ) :
         
        X[:, 1:] = ( X[:, 1:] - np.mean( X[:, 1:], axis = 0 ) ) / np.std( X[:, 1:], axis = 0 )
         
        return X
         
    # model training
     
    def fit( self, X, Y ) :
         
        self.X = X
     
        self.Y = Y
     
        self.m, self.n = self.X.shape
     
        # weight initialization
     
        self.W = np.zeros( self.degree + 1 )
         
        # transform X for polynomial  h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n
         
        X_transform = self.transform( self.X )
         
        # normalize X_transform
         
        X_normalize = self.normalize( X_transform )
                 
        # gradient descent learning
     
        for i in range( self.iterations ) :
             
            h = self.predict( self.X )
         
            error = h - self.Y
             
            # update weights
         
            self.W = self.W - self.learning_rate * ( 1 / self.m ) * np.dot( X_normalize.T, error )
         
        return self
     
    # predict
     
    def predict( self, X ) :
      
        # transform X for polynomial  h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n
         
        X_transform = self.transform( X )
         
        X_normalize = self.normalize( X_transform )
         
        return np.dot( X_transform, self.W )
       
       
# Driver code    
 
def main() :   
     
    # Create dataset
     
    
  
    # model training
     
    model = PolynomailRegression( degree = 2, learning_rate = 0.01, iterations = 500 )
 
    model.fit( x, y )
     
    # Prediction on training set
 
    Y_pred = model.predict( x )
     
    # Visualization
     
    plt.scatter( x, y, color = 'blue' )
     
    plt.plot( x, Y_pred, color = 'orange' )
     
    plt.title( 'X vs Y' )
     
    plt.xlabel( 'X' )
     
    plt.ylabel( 'Y' )
     
    plt.show()

main()



""" -------------------------------------------------------------------------------"""
""" RNN (Recurrent Neural Network) from scratch"""
""" -------------------------------------------------------------------------------"""
"""
# Load the data and create training and testing data
#X_train,y_train = x,y
#X_test,y_test = x,y
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 100)

# Building and training our model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Making predictions with our data
predictions = clf.predict(X_test)
print(predictions[:5])

# Visualizing the linear function for our SVM classifier
w = clf.coef_[0]
b = clf.intercept_[0]
x_visual = np.linspace(32,57)
y_visual = -(w[0] / w[1]) * x_visual - b / w[1]

scatterplot(data = X_train, x='bill_length_mm', y='bill_depth_mm', hue=y_train)
plt.plot(x_visual, y_visual)
plt.show()

# Testing the accuracy of our model
"""