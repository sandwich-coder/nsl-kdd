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


# - situation -

#check in 10 days first and extend on success
result['short']['success'] = - 6 * 10 - 70 * 4
result['short']['fail'] = - 6 * 10    #out two days after the announcement

#check in 1 month first and extend on success
result['long1']['success'] = - 90 * 1 - 75 * 3
result['long1']['fail'] = - 90 * 1    #out in a month
temp = - 6 * 10 - 30    #out two days after the announcement | small discount applied to unintended extensions as well
if result['long1']['fail'] < temp:
    result['long1']['fail'] = temp

#check in 4 months and remain on success
result['long2']['success'] = - 70 * 4
result['long2']['fail'] = - 90 * 1 - 30    #out in a month
temp = - 6 * 10 - 30    #out two days after the announcement
if result['long2']['fail'] < temp:
    result['long2']['fail'] = temp

print('short: {}'.format(result['short']))
print('long1: {}'.format(result['long1']))
print('long2: {}'.format(result['long2']))
print('\n')


# - expected flow -

expected = {}

print('''\
                Expected Cash Flow
          ------------------------
Chance ||  short |  long1 |  long2
------ || ------ | ------ | ------\
    ''')
chance = np.linspace(np.float64(0), np.float64(1), num = 100, endpoint = False)
chance = chance.tolist()
for l in chance:

    #expected values
    expected['short'] = round(result['short']['success'] * l + result['short']['fail'] * (1 - l), ndigits = 1)
    expected['long1'] = round(result['long1']['success'] * l + result['long1']['fail'] * (1 - l), ndigits = 1)
    expected['long2'] = round(result['long2']['success'] * l + result['long2']['fail'] * (1 - l), ndigits = 1)

    #most efficient option
    cheap = max(*expected.values())
    cheap = list(expected.values()).index(cheap)
    cheap = list(expected.keys())[cheap]

    print('{:>5}% || {:>6} | {:>6} | {:>6}   -> cheapest choice :  {}'.format(
        round(l * 100, ndigits = 1),
        expected['short'], expected['long1'], expected['long2'],
        cheap,
        ))
