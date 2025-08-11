from copy import deepcopy as copy
import inspect, code
import types
import time
import numpy as np
from scipy import linalg as la


result = {}
result['short'] = {}
result['long1'] = {}
result['long2'] = {}


#check in 9 days first and extend on success
result['short']['success'] = - 6 * 9 - 70 * 4
result['short']['fail'] = - 6 * 9    #out in 9 days

#check in 1 month first and extend on success
result['long1']['success'] = - 90 * 1 - 75 * 3
result['long1']['fail'] = - 90 * 1    #out in a month
temp = - 7 * 9 - 30    #out in 9 days
if result['long1']['fail'] < temp:
    result['long1']['fail'] = temp

#check in 4 months and remain on success
result['long2']['success'] = - 70 * 4
result['long2']['fail'] = - 90 * 1 - 30    #out in a month
temp = - 7 * 9 - 30    #out in 9 days
if result['long2']['fail'] < temp:
    result['long2']['fail'] = temp

print('short: {}'.format(result['short']))
print('long1: {}'.format(result['long1']))
print('long2: {}'.format(result['long2']))
print('\n')


expected = {}

print('                Expected Cash Flow')
print('          ------------------------')
print('Chance ||  short |  long1 |  long2')
print('------ || ------ | ------ | ------')
chance = np.linspace(np.float64(0), np.float64(1), num = 1000, endpoint = False)
chance = chance.tolist()
for l in chance:
    expected['short'] = round(result['short']['success'] * l + result['short']['fail'] * (1 - l), ndigits = 1)
    expected['long1'] = round(result['long1']['success'] * l + result['long1']['fail'] * (1 - l), ndigits = 1)
    expected['long2'] = round(result['long2']['success'] * l + result['long2']['fail'] * (1 - l), ndigits = 1)

    most_saving = max(*expected.values())
    efficient = list(expected.values()).index(most_saving)
    efficient = list(expected.keys())[efficient]
    print('{:>5}% || {:>6} | {:>6} | {:>6}   -> best choice :  {}'.format(
        round(l * 100, ndigits = 1),
        expected['short'],
        expected['long1'],
        expected['long2'],
        efficient,
        ))
