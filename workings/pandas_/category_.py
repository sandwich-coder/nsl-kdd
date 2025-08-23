import pandas as pd


df = pd.DataFrame({
    'letter':['a', 'b', 'c', 'd', 'e', 'f'],
    })
df = df.astype('category', copy = False)


first = df.iloc[:3, :].copy()    #first three
second = df.iloc[3:, :].copy()    #last three

first_onehot = pd.get_dummies(first, columns = ['letter'])
second_onehot = pd.get_dummies(second, columns = ['letter'])


# The categories that don't even exist are listed by the 'value_counts', with the counts 0.
# Furthermore, the 'get_dummies' "onehots" based not on what actually are but rather on what are listed in the category dtype.
# What actually are affect not the onehot behavior but the initial conversion to the catogory dtype.

print(first['letter'].value_counts())
print('')
print(first_onehot)
print('\n')

print(second['letter'].value_counts())
print('')
print(second_onehot)
