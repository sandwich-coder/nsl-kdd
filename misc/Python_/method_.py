from types import MethodType


# - function -

class Foo:
    def __init__(self):
        def func():
            pass
        self.bar = func

foo = Foo()
foo_bar1 = foo.bar
foo_bar2 = foo.bar
print(foo_bar1 is foo_bar2)
print(type(foo.bar))


print('')

# - method -

class Foo:
    def __init__(self):
        pass
    def bar(self):
        pass

foo = Foo()
foo_bar1 = foo.bar
foo_bar2 = foo.bar
print(foo_bar1 is foo_bar2)    # The differing of addresses seems to come from the instance generation of MethodType.
print(type(foo.bar))
print(type(Foo.bar))    # 'Foo.bar' is not some "fallback option" of 'foo.bar'. Rather, python automatically GENERATES new methods based on the predefined functions of that class. 'foo.bar' is thus an instance attribute, which originated from class attribute 'Foo.bar'.


"""
The first input of a function can be "pushed inside" to be included in the definition.
Such "reduced" function is called a 'method' of the original. In fact,
Python offers tools to make methods manually.
"""

def add(a, b):
    return a + b

temp = 1
exec("add_to_{} = MethodType(add, temp)".format(temp))
