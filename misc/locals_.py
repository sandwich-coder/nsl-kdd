import inspect


def foo():
    a = 0
    local_names = locals()

    print(local_names['a'])

    #read
    a = 1
    print(local_names['a'])
    a = 0

    #write
    local_names['a'] = 1
    print(a)
    local_names['a'] = 0


def bar():
    a = 0
    frame = inspect.currentframe()

    print(frame.f_locals['a'])

    #read
    a = 1
    print(frame.f_locals['a'])
    a = 0

    #write
    frame.f_locals['a'] = 1
    print(a)
    frame.f_locals['a'] = 0


print('\n')
foo()
print('\n')
bar()
print('\n')


"""
'inspect.currentframe' returns a live frame like 'globals', as opposed to 'locals' which returns a snapshot of the moment.
However, only the 'f_locals' property returns a copy of the dictionary that is updated on read but not write-through,
inconsistent with being a live object. This is fixed in Python3.13, 'inspect.currentframe' becoming fully live.
"""
