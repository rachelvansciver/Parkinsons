'''
Created on Jun 29, 2020
@author: Rachel
dataset downloadable at https://archive.ics.uci.edu/ml/datasets/Parkinsons
Source cited from dataset 
'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', 
Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. BioMedical Engineering OnLine 2007,
6:23 (26 June 2007)
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

raw_df = pd.read_csv("D:/Downloads/parkinsons.data");
#raw data from file
num_data = raw_df.loc[:, raw_df.columns != 'status'].values[:,1:]
#numerical data to be normalized, doesn't include the status or name columns
hasParkinsons = raw_df.loc[:, 'status'].values
#binary attribute column - 0 = healthy, 1 = has parkinsons
scale = MinMaxScaler((-1,1))
#transforms data to be on the scale of -1 to 1
norm_data = scale.fit_transform(num_data)
#normalized data
x_train, x_test, y_train, y_test = train_test_split(norm_data, hasParkinsons,test_size = 0.25, random_state = 42, shuffle=True)
#splits data into testing and training sets

regres = LogisticRegression()
#initializing regression obj
regres = regres.fit(x_train, y_train)
#fitting data to model

y_pred = regres.predict(x_test)
#predicting y based on x test values
fig, ax = plt.subplots()
print(regres.score(x_test, y_test))
#model is accurate, has score of 0.8979591836734694
plt.show()


