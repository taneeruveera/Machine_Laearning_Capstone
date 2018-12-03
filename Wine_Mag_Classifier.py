import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import nltk

data = pd.read_csv("C:\winemag-data_first150k.csv")
print("Length of dataframe before duplicates are removed:", len(data))
data = data.drop_duplicates('description')
data = data[pd.notnull(data.price)]
data = data[pd.notnull(data.variety)]
#print(data.shape)
#print(data.head)
value_counts = data['country'].value_counts()
data = data[~data['country'].isin(value_counts[value_counts < 500].index)]
#print(data.country.value_counts())
grape_counts = data['variety'].value_counts()
data = data[~data['variety'].isin(grape_counts[grape_counts < 500].index)]
#print(data.variety.value_counts())
print(data.shape)
sw = ['pinot', 'noir', 'chardonnay', 'cabernet', 'sauvignon', 'blanc', 'syrah', 'riesling', 'bordeaux', 'merlot', 'zinfandel', 'malbec', 'sangiovese', 'rose', 'tempranillo', 'shiraz', 'sparkling', 'portuguese', 'rhone', 'nebbiolo', 'corvino', 'rondinella', 'molinara', 'viognier', 'cabaret', 'franc', 'gris', 'grigio', 'champagne', 'grosso', 'gewurztraminer', 'petite', 'sirah', 'gruner', 'veltliner', 'port']
print(sw)
def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space,
    # which in effect deletes the punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)
data['description'] = data['description'].apply(remove_punctuation)
#temp_desc[i] = [word for word in temp_desc[i] if word not in stop_words]
def stopwords(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [ word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)
#print(data.description[:1])
data['description'] = data['description'].apply(stopwords)

col = ['country', 'description', 'designation', 'points', 'price', 'province', 'region_1', 'region_2', 'variety', 'winery']
data = data.replace(np.nan, 0)
print(data.dtypes)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

X = data[['country', 'description', 'designation', 'points', 'price', 'province', 'region_1', 'region_2', 'variety', 'winery']]
y=data.variety

y=labelencoder.fit_transform(y)
data = pd.DataFrame(y)
#print(y.shape)
#data.to_csv("F:\winemag-data_Clean.csv")


import category_encoders as ce
targetencoder = ce.TargetEncoder(cols=['country', 'description', 'designation', 'province', 'price', 'points', 'region_1', 'region_2', 'winery'])
targetencoder.fit(X, y)
X_Encoded = targetencoder.transform(X)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X_Encoded, y, random_state=1, test_size=0.2)
#data = pd.DataFrame(y)
#data.to_csv("F:\winemag-data_Clean.csv")
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

accuracy = accuracy_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred, average='weighted')
print("score for decision tree classifier")
print(accuracy, F1_score)
'''
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=31, random_state=0)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
accuracy1 = accuracy_score(y_test,y_pred)
F1_score1 = f1_score(y_test, y_pred, average='weighted')
print("score for Random Forrest classifier")
print(accuracy1, F1_score1)

from sklearn.ensemble import AdaBoostClassifier
model2 = AdaBoostClassifier(n_estimators=50, learning_rate = 0.01, random_state=0)
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
accuracy2 = accuracy_score(y_test,y_pred)
F1_score2 = f1_score(y_test, y_pred, average='weighted')
print("score for Adaboost classifier")
print(accuracy2, F1_score2)

from sklearn.svm import SVC
model3 = SVC(gamma='auto')
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)
accuracy3 = accuracy_score(y_test,y_pred)
F1_score3 = f1_score(y_test, y_pred, average='weighted')
print("score for SVM classifier")
print(accuracy3, F1_score3)


from lightgbm import LGBMClassifier
model4 = LGBMClassifier(objective='multiclass', random_state=5)
model4.fit(X_train, y_train)
y_pred = model4.predict(X_test)
accuracy4 = accuracy_score(y_test,y_pred)
F1_score4 = f1_score(y_test, y_pred, average='weighted')
print("score for lightGBM classifier")
print(accuracy4, F1_score4)

from xgboost import XGBClassifier
model5 = XGBClassifier()
model5.fit(X_train, y_train)
y_pred = model5.predict(X_test)
accuracy5 = accuracy_score(y_test,y_pred)
F1_score5 = f1_score(y_test, y_pred, average='weighted')
print("score for lightGBM classifier")
print(accuracy5, F1_score5)

'''