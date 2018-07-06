import pandas as pd
import numpy as np
from sklearn import preprocessing

# reading data
train = pd.read_csv("F:\\study\\datasets\\tit\\train.csv")
test = pd.read_csv("F:\\study\\datasets\\tit\\test.csv")


def get_object(df):
    obj = []
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            obj.append(col)
    return obj


def get_naData(df):
    na = []
    for col in df.columns:
        if df[col].isnull().any():
            na.append(col)
    return na


# preparing the data
# NAME column
def get_surname(df):
    Surname = []
    for i in range(len(df.Name)):
        string = df.Name.iloc[i]
        surname = ''
        dotp = string.find('.')
        for i in range(dotp, 0, -1):
            if string[i] == ' ':
                break
            surname += string[i]
        surname = surname[::-1]
        if surname == '.':
            surname = None
        elif surname == 'rs.':
            surname = 'Mrs.'
        elif surname == 'r.':
            surname = 'Mr.'
        elif surname == 'iss.':
            surname = 'Miss.'
        elif surname == 'aster.':
            surname = 'Master.'
        Surname.append(surname)
    return pd.Series(Surname)


# label encoding
def encode(ser):
    lab_enc = preprocessing.LabelEncoder()
    return lab_enc.fit_transform(ser.values.astype(str))


# filling data
def fill(ser):
    imp = preprocessing.Imputer('NaN', strategy='mean')
    return imp.fit_transform(ser.values.reshape(-1, 1))
