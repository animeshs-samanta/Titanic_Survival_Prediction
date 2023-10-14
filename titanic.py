# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 23:31:32 2023

@author: Animesh
"""

# Titanic Survival Prediction

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")
%matplotlib inline
warnings.filterwarnings("ignore")



#Read the data set, Traning dataset , Test dataset

train_data = pd.read_csv(r"A:\Bharat Intern\Titanic survival prediction\train.csv")
test_data = pd.read_csv(r"A:\Bharat Intern\Titanic survival prediction\test.csv")

train_data.shape

train_data.columns.values

train_data.info()

train_data.describe()

# Above we can see that 38% out of the training-set survived the Titanic. We can also see that the passenger ages range from 0.4 to 80.
# Preview the data

train_data.head(10)

# Numerical and alpha numeric data within the same feature, Ticket is mix of numeric and alphanumeric data types, Cabin is alphanumeric

train_data.tail(10)


train_data.isnull()

train_data.isnull().sum()

# Cabin, Age,Embarked features contain a number of null values 


f, ax = plt.subplots(1, 2, figsize=(12, 4))
train_data['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=False)
ax[0].set_title('Survivors (1) and Not Survivors (0)')
ax[0].set_ylabel('')

sns.countplot(x='Survived', data=train_data, ax=ax[1])  # Specify 'x' to represent the data column
ax[1].set_ylabel('Quantity')
ax[1].set_title('Survivors (1) and Not Survivors (0)')

plt.show()


f, ax = plt.subplots(1, 2, figsize=(12, 5)) 
train_data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0]) 
ax[0].set_title('Survivors by sex') 
sns.countplot(x='Sex', hue='Survived', data=train_data, ax=ax[1]) 
ax[1].set_ylabel('Quantity') 
ax[1].set_title('Survived (1) and death (0): men and women') 
plt.show()



# ---------------------Data Preprocessing--------------------------


#Here we drop unnecessary feature and convert string data into numerical data for easier training.

#e will drop the the Cabin feature since not a lot more useful information can be extracted from it

# Create a new column cabin_New indicating 
# if the cabin value was given or was NaN 

train_data["Cabin_new"] = (train_data["Cabin"].notnull().astype("int"))
test_data["Cabin_new"] = (test_data["Cabin"].notnull().astype("int"))

#Delete the column "Cabin" from test and triaing data set

train_data = train_data.drop(["Cabin"], axis = 1)
test_data = test_data.drop(["Cabin"], axis = 1)




#We can also drop the Ticket feature since it’s unlikely to yield any useful information

train_data = train_data.drop(['Ticket'], axis=1) 
test_data = test_data.drop(['Ticket'], axis=1) 



# There are missing values in the Embarked feature. For that, we will replace the NULL 
# values with ‘S’ as the number of Embarks for ‘S’ are higher than the other two.

train_data = train_data.fillna({"Embarked" : "S"})


#We will now sort the age into groups. We will combine the age groups of the people and categorize them into the same groups.
#BY doing so we will be having fewer categories and will have a better prediction since it will be a categorical dataset.

train_data["Age"] = train_data["Age"].fillna(-0.5)
test_data["Age"] = test_data["Age"].fillna(-0.5)
                                                                                                                                        
bins = [-1,0,5,12,18,24,35,60, np.inf]
labels = ["Unknown" , "Baby" , "Child" , "TeenAge" , "Student" , "Young Adult" , "Adult" , "Senoir"]

train_data["AgeGroup"] = pd.cut(train_data["Age"],bins, labels=labels)
test_data["AgeGroup"] = pd.cut(test_data["Age"],bins, labels=labels)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
train_data['AgeGroup'] = label_encoder.fit_transform(train_data['AgeGroup'])
test_data['AgeGroup'] = label_encoder.transform(test_data['AgeGroup'])



train_data.tail(10)


#combine the train and test dataset
combine = [train_data, test_data]

#Extract the title for each name in tarin and test dataset
for dataset in combine:
    dataset["Title"] = dataset.Name.str.extract("([A-Za-z]+)\.", expand = False)
    
pd.crosstab(train_data["Title"], train_data["Sex"])




#We can replace many titles with a more common name or classify them as Rare

for dataset in combine:
    dataset["Title"] = dataset["Title"].replace(["Lady","Capt","Col","Don","Dr","Major","Rev","Jonkheer","Dona"],"Rare")
    
    dataset["Title"] = dataset["Title"].replace(["Countess","Lady","Sir"],"Royal")
    dataset["Title"] = dataset["Title"].replace("Mlle","Miss")
    dataset["Title"] = dataset["Title"].replace("Ms","Miss")
    dataset["Title"] = dataset["Title"].replace("Mme","Mrs")
    
train_data[["Title", "Survived"]].groupby(["Title"], as_index = False).mean()



# We can convert the categorical titles to a numerical value.

title_mapping = {"Mr" : 1, "Miss" : 2, "Mrs" : 3, "Master" : 4, "Royal" : 5, "Rare" : 6}
for dataset in combine:
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0)
    
train_data.head()



# Now we can drop the Name feature from traing and testing datasets, We also do not need the passenger

train_data = train_data.drop(["Name"], axis =1)
test_data = test_data.drop(["Name"], axis =1)

combine = [train_data, test_data]


# Let us start by converting Sex feature to a new feature called Gender where female=1 and male=0.

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_data.head()


'''
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 1, 'C': 2, 'Q': 3} ).astype(int)

''' 
                                                                                                                          
embarked_mapping = {'S': 1, 'C': 2, 'Q': 3}

train_data["Embarked"] = train_data["Embarked"].map(embarked_mapping)
test_data["Embarked"] = test_data["Embarked"].map(embarked_mapping)

train_data.head()



#Fill in the missing Fare value in the test set based on the mean fare for that P-class

for i in range(len(test_data["Fare"])):
    if pd.isnull(test_data["Fare"][i]):
        pclass = test_data["Pclass"][i] 
        test_data["Fare"][i] = round(train_data[train_data["Pclass"] == pclass] ["Fare"].mean(),4)




#map Fare values into groups of numerical values

train_data["FareBand"] = pd.qcut(train_data["Fare"],4, labels = [1,2,3,4])

test_data["FareBand"] = pd.qcut(test_data["Fare"],4, labels = [1,2,3,4])



#drop the Fare values

train_data = train_data.drop(["Fare"], axis = 1)

test_data = test_data.drop(["Fare"], axis = 1)

train_data.head(10)


#Model Training

from sklearn.model_selection import train_test_split 

# Drop the Survived and PassengerId 
# column from the trainset 
predictors = train_data.drop(['Survived', 'PassengerId'], axis=1) 
target = train_data["Survived"] 
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=0) 



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Fit the traing the data
randomforest = RandomForestClassifier(n_estimators =100)
randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_test)

accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
print("Accuracy is : ",accuracy)



# Confusion Matrix

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
prediction = cross_val_predict (randomforest, x_train, y_train, cv = 3)
cm = confusion_matrix(y_train, prediction)

print("Confusion Matrix :\n ",cm)
# Precision and Recall

from sklearn.metrics import precision_score, recall_score

print("Precision : " ,precision_score(y_train, prediction))

print("Recall : ", recall_score(y_train, prediction))

  
# F-Score

from sklearn.metrics import f1_score
f1 = f1_score(y_train, prediction)

print("F1-Score is :", f1)

