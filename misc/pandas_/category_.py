import pandas as pd


df = pd.DataFrame({
    'letter':['a', 'b', 'c', 'd', 'e', 'f'],
    })
df = df.astype('category', copy = False)


first = df.iloc[:3, 0].copy()    #first three
second = df.iloc[3:, 0].copy()    #last three

first_onehot = pd.get_dummies(first, columns = ['letter'])
second_onehot = pd.get_dummies(second, columns = ['letter'])


print(first_onehot)
print('')
print(second_onehot)
