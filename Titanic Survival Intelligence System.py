from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
def loadData():
    return pd.read_csv('titanic.csv')

def preProcessing(df):
    df= df[['Survived','Pclass','Sex','Age','SibSp','Parch','Name']]
    df['Sex']= df['Sex'].map({'male':0, 'female':1})


    return df

def fe(df):
    df['FamilySize']= df['SibSp'] + df['Parch']+1
    df['IsAlone'] =  (df['FamilySize']==1).astype(int)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df= pd.get_dummies(df, columns=['Title'])
    return df


def train(df):
    X= df.select_dtypes(include= ['int64','float64']).drop(['Survived'])
    y= df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_test, y_test

def evulation(model, X_test, y_test):
    return  model.score(X_test, y_test)



df = loadData()

df = preProcessing(df)

df = fe(df)

model, X_test, y_test = train(df)

accuracy = evaluation(model, X_test, y_test)

print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")



