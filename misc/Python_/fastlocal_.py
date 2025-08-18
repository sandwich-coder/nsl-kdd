def foo():
    exec('a = 1')    #update 'locals'
    print(a)    #seems to load from the fast storage

def bar():
    exec('a = 1')
    exec('print(a)')    #seems to load from 'locals()'


try:
    foo()
except NameError:
    print('names not recognized')
except:
    print('errors ignored')

try:
    bar()
except NameError:
    print('names not recognized')
except:
    print('errors ignored')
