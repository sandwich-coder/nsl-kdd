import inspect


def foo():
    local_names = locals()
    frame = inspect.currentframe()

    print('--- Before ---')
    print('local_names.get(\'a\'): {}'.format(local_names.get('a')))
    print('locals().get(\'a\'): {}'.format(locals().get('a')))
    print('frame.f_locals.get(\'a\'): {}'.format(frame.f_locals.get('a')))
    print('')

    a = 1
    print('--- Modified Outside < a = 1 > ---')
    print('local_names.get(\'a\'): {}'.format(local_names.get('a')))
    print('locals().get(\'a\'): {}'.format(locals().get('a')))
    print('frame.f_locals.get(\'a\'): {}'.format(frame.f_locals.get('a')))
    print('')

    local_names['a'] = 2
    print('--- Locals Modified < a = 2 > ---')
    print('a:', a)
    print('a:', a)
    print('')

    frame.f_locals['a'] = 3
    print('--- Frame Modified < a = 3 > ---')
    print('a:', a)
    print('a:', a)


foo()


"""
The above mismatching behavior of the 'frame.f_locals'
between read and write is fixed in Python3.13.
"""
