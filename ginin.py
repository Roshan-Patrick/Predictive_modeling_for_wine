# Importing the required packages 
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

balance_data = pd.read_csv('D:/ML/DATASET/winequality-red.csv') 
# Printing the dataswet shape 
print ("Dataset Length: ", len(balance_data)) 
print ("Dataset Shape: ", balance_data.shape) 
	
# Printing the dataset obseravtions 
print ("Dataset: ",balance_data.head()) 
X = balance_data.values[:, 0:11] 
Y = balance_data.values[:, 11] 	

# Spliting the dataset into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) 

#Creating a gini based model
model = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 

# Actual training process 
model.fit(X_train, y_train) 

# Predicton on testdata 
y_pred = model.predict(X_test) 
print("Predicted values:") 
print(y_pred)

print("Confusion Matrix:\n ", confusion_matrix(y_test, y_pred)) 
	
print ("Accuracy : ", accuracy_score(y_test,y_pred)*100) 
	
print("Report : ", classification_report(y_test, y_pred))
	