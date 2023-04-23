import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('healthcare-dataset-stroke-data.csv') # read data

# Α. Ανάλυση του Dataset

print(df.info(),'\n')
#infos=df.describe()
#print(infos.drop('stroke',axis=1))
#infos2=df.describe(include='object')
#print(infos2)

for col in df.columns:
    if (df[col].values.dtype=='O'): # strings
        print("Column:",col)
        print(df[col].describe())
        print("Unique values: ",df[col].unique(),'\n')
    elif (np.array_equal(df[col].unique(),[0,1]) or np.array_equal(df[col].unique(),[1,0]) ): # 0 or 1
        print("Column:",col)
        print("Count: ",df[col].count())
        print("value | times")
        print(df[col].value_counts(),'\n')
        #print("Unique values: ",df[col].unique(),'\n')
    else: # numbers
        print("Column:",col)
        print(df[col].describe().drop(['std','25%','50%','75%'],axis=0),'\n')
        
#-- γραφική αναπαράσταση του ποσοστού ύπαρξης εγκεφαλικού --

#y=np.array([df[df["stroke"]==0].count().stroke,df[df["stroke"]==1].count().stroke])
y2=[df[df["stroke"]==1].count().stroke,df[df["stroke"]==0].count().stroke]
plt.pie(y2,labels=["Yes","No"],colors=["r","g"],explode=[0.1,0],autopct="%.2f")
plt.title("Stroke")
plt.show()

# Β. Ελλιπείς Τιμές

# 0. Προεπεξεργασία τιμών
df.replace('Unknown', np.NaN, inplace=True)
df.hypertension.replace([0, 1], ['No', 'Yes'], inplace=True)
df.heart_disease.replace([0, 1], ['No', 'Yes'], inplace=True)
df.stroke.replace([0, 1], ['No', 'Yes'], inplace=True)

# 1. Αφαίρεση στήλης
df1=df.copy()
for col in df1.columns:
    if df1[col].isnull().values.any(): # detect column with missing values
        df1.drop(col,axis=1,inplace=True)
        #print(col)

# 2. Μέσος όρος
df2=df.copy()
for col in df2.columns:
    if df2[col].isnull().values.any(): # detect column with missing values
        if (df2[col].values.dtype=='O'): # strings
            df2[col].fillna(df2[col].mode()[0],inplace=True)
            #print(col)
        else:
            df2[col].fillna(df2[col].mean(),inplace=True)
            #print(col) 

# 3. Linear Regression 
df3=df.copy()
df3.interpolate('linear', inplace=True)

# 4. k-Nearest Neighbors



# Γ. Random Forest (πρόβλεψη)
def RandomForest(dataset,colin):
    dataset.gender = LabelEncoder().fit_transform(dataset['gender'])
    dataset.hypertension = LabelEncoder().fit_transform(dataset['hypertension'])
    dataset.heart_disease = LabelEncoder().fit_transform(dataset['heart_disease'])
    dataset.ever_married = LabelEncoder().fit_transform(dataset['ever_married'])
    dataset.work_type = LabelEncoder().fit_transform(dataset['work_type'])
    dataset.Residence_type = LabelEncoder().fit_transform(dataset['Residence_type'])
    if colin:
        dataset.smoking_status = LabelEncoder().fit_transform(dataset['smoking_status'])
    dataset.stroke = LabelEncoder().fit_transform(dataset['stroke'])

    x=dataset.drop('stroke',axis=1)
    y=dataset.stroke

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print('\tf1_score: {0:.2f}%,\tprecision: {1:.2f}%,\trecall: {2:.2f}%'.format(
        f1_score(y_test, y_pred, average='weighted') * 100,
        precision_score(y_test, y_pred, average='weighted') * 100,
        recall_score(y_test, y_pred, average='weighted') * 100))

print("Remove collumns with empty values")
RandomForest(df1,False)
print("Replace empty values with column mean")
RandomForest(df2,True)
print("Replace empty values with Linear Regression")
RandomForest(df3,True)
print("Replace empty values with k-Nearest Neighbours")
#RandomForest(df4,True)
