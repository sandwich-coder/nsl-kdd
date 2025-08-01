
#global scope
exec('a = 1')
print(locals().get('a'))    # The 'locals' returns the snapshot, EXCEPT in the global scope.

def foo():

    #local scope
    exec('b = 1')
    print(locals().get('a'))

foo()
