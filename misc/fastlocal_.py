def foo():
    exec('a = 1')
    print(a)

def bar():
    exec('a = 1')
    exec('print(a)')


try:
    foo()
except NameError:
    print('name not recognized')
except:
    print('error ignored')

try:
    bar()
except NameError:
    print('name not recognized')
except:
    print('error ignored')
