# - global scope -

exec('a = 1')

#natural
try:
    print(a)
except NameError:
    print('It is not visible.')
except:
    print('errors ignored')

#from dictionary
try:
    print(locals()['a'])
except KeyError:
    print('It is not in the dictionary.')
except:
    print('errors ignored')


print('')

# - local scope -

def foo():
    exec('b = 1')

    #natural
    try:
        print(b)
    except NameError:
        print('It is not visible.')
    except:
        print('errors ignored')

    #from dictionary
    try:
        print(locals()['b'])
    except KeyError:
        print('It is not in the dictionary.')
    except:
        print('errors ignored')

foo()



"""
Function 'exec' doesn't behave as expected in the local scope.
The reason is not in some peculiarity of the function per se,
but in the peculiar mechanism of how local names are stored.
"""
