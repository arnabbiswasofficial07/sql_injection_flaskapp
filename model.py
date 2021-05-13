from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
from sklearn.utils import shuffle
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

sql = pd.read_csv('sql.csv')
password = pd.read_csv('password.csv')
username = pd.read_csv('username.csv')
sqli = pd.read_csv('sqli.csv')

username.dropna(axis=0, how="all", inplace=True)
password.dropna(axis=0, how="all", inplace=True)
username.reset_index(drop=True, inplace=True)
password.reset_index(drop=True, inplace=True)


def leng(df, col, len_col):
    for i in range(len(df)):
        cl = df[col][i]
        length = len(str(cl))
        df[len_col][i] = length
    return df


username = leng(username, 'Query', 'Length')
password = leng(password, 'Query', 'Length')
sql = leng(sql, 'Query', 'Length')

username['Label'] = 'username'
password['Label'] = 'password'
sqli['Label'] = 'sqli'
sql['Label'] = 'sql'

sqli.drop(['Attack'], axis=1, inplace=True)
username.drop(['Attack'], axis=1, inplace=True)
password.drop(['Attack'], axis=1, inplace=True)
sql.drop(['Attack'], axis=1, inplace=True)

df = pd.concat([sqli, sql, username, password])
df.reset_index(drop=True, inplace=True)


def cal_puncndop(df, col, punop_col, l):
    df1 = df[[col]].copy()
    for i, query in enumerate(df[col]):
        count = 0
        li = list(query)
        for ch in range(len(query)):
            if query[ch] in l:
                li[ch] = " "
                count = count + 1
        df1[col][i] = "".join(li)
        # print(("".join(li)))
        df[punop_col][i] = count
    df[col] = df1[col]
    return df


df['punctuation'] = 0
df = cal_puncndop(df, 'Query', 'punctuation', ["<",">", "<=", ">=", "=", "==", "!=", "<<", ">>", "|", "&", "-", "+", "%", "^", "*"])


def cal_keyword(df, col, key_col, l):
    for i, query in enumerate(df[col]):
        count = 0
        query = query.lower()
        words = query.split()
        for word in words:
            if word in l:
                count = count + 1
        df[key_col][i] = count
    return df


df['keyword'] = 0
df = cal_keyword(df, 'Query', 'keyword',
                 ["select", "update", "insert", "create", "drop", "alter", "rename", "exec", "order", "group", "sleep","count", "where"])

for i, label in enumerate(df['Label']):
    if label in ['sql', 'username', 'password']:
        df['Label'][i] = 'non-sqli'

df['Label'] = df['Label'].replace(['sqli'], 1)

df['Label'] = df['Label'].replace(['non-sqli'], 0)

df = shuffle(df)

X = df.drop(labels=['Label', 'Query'], axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

# Import svm model
# Create a svm Classifier
clf = svm.SVC(kernel='linear')  # Linear Kernel

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))

pickle.dump(clf, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

print(model.predict([[115, 9, 3]]))
