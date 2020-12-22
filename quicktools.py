'''
Here is the documentation of quicktools module.

(You could skip the following one paragraph and go straight
to the part under the split lines "======")

Before anything, actually I thought for a while how to get the local and
global variables from the file that imports this module so that this module
could use those variables when using test function and case function (I will
introduce them in a while), in other words, I tried to find a way to make
the file that imports this module could share with this module. I will
talks about the reasons later. After googling a lot, I still cannot find
a suitable method to achieve this. Finally, I came up with a proper solution,
at least for my own situation. I define two global variables inside this module,
which are LOC and GLOB, users could use a function called "update" to fill
the locals() and globals() from the file imports this module into this module.
When any places in this module uses the eval() or exec(), it will use LOC
and GLOB instead of locals() and globals() of this module itself.
This method has great feasibility, after a lot of tests by me.
=============================================================================

The module quicktools is designed for simplifying programming steps and to relief
the pressure of programmers when they are forced to write a bunch of nested
loops, nested multiple if-elif-else statements, get multiple elements through
discontinous indexes, need to make some special functional stuffs but 
the built-in functions only supports limited features, and so on.
I will go through some of the most important functions in this module,
for others, please check the function documentations for them.

1. upate(loc, glob)
This is the very first function you need to use when you import quicktools.
The step is pretty easy, just within half of a line, now assuming
you import this module like: import quicktools as qt
then you just write "qt.update(locals(), globals())".
Once you include this line in you file, then any variables in you file
would be shared with this module. Even if you change your variables during
the whole process in your file, the variables that shares with this module
would dynamically change simultaneously. It is required to include this line
after you import quicktools, otherwise the variables in your file cannot
share with this module (which means this module cannot use any of the
variables in your file)

2. test(program, digit = 3, output = True, printed = False):
This is a test function for getting the processing time of functions or
any executable statements. The variable program is a string that contains
the stuff you want to execute, digit is the number of the decimals after the time,
and you can decide this function to print the output or not by setting
otuput to True or False. The return of this function is formatted as
hours, minutes, seconds, result is ... (if output is set to True)

3. mult(*obj):
This is a function that do things very similar to sum() but replace the
add with multiply. You need to ensure that all of the elements inside the
list are multipliable. This function might be very useful in mathematical
and physical calculations, such as matrices, sequences.




'''
import time, fractions, datetime, os
from copy import deepcopy as cp

LOC, GLOB = locals(), globals()


def update(loc, glob):
    global LOC
    global GLOB
    LOC, GLOB = loc, glob


def rounding(num, dec=None, tostr=False):
    if dec is None or isinstance(num, int):
        return num
    if isinstance(num, fractions.Fraction):
        num = float(num)
    numstr = str(num)
    ind = numstr.index('.')
    if dec == 0:
        intpart = eval(numstr[:ind], LOC, GLOB)
        return intpart + 1 if eval(numstr[ind +
                                          1], LOC, GLOB) >= 5 else intpart
    tol = len(numstr) - ind - 1
    if tol < dec:
        return f'{num:.{dec}f}' if tostr else num
    elif tol == dec:
        return num
    else:
        if eval(numstr[ind + dec + 1], LOC, GLOB) >= 5:
            temp = str(num + 10**-dec)[:ind + dec + 1]
            result = eval(temp, LOC, GLOB)
            resultstr = str(result)
            if len(resultstr) - resultstr.index('.') - 1 == dec:
                return result
            else:
                return f'{result:.{dec}f}' if tostr else result

        else:
            return eval(numstr[:ind + dec + 1], LOC, GLOB)


fractcheck = lambda x: x if x != 0 else 0


def formating(a):
    for i in range(len(a)):
        current = a[i]
        if type(current) in [float, fractions.Fraction]:
            y = fractions.Fraction(current).limit_denominator()
            a[i] = int(y) if y.__float__().is_integer() else y
    return a


def test(program, digit=3, output=True, printed=False):
    start = time.time()
    result = eval(program, LOC, GLOB)
    stop = time.time()
    spend = stop - start
    second = rounding(spend % 60, digit)
    minute = int(spend // 60)
    hour = int(spend // 3600)
    if output:
        if not printed:
            return f'cost time: {hour}h {minute}m {second}s, result is {result}'
        else:
            print(f'cost time: {hour}h {minute}m {second}s')
            print('result is')
            print(result)
            return
    return f'cost time: {hour}h {minute}m {second}s'


def sums(*obj):
    if isinstance(obj[0], list):
        obj = obj[0]
        if type(obj[0]) in [int, float, fractions.Fraction, complex]:
            return sum(obj)
        temp = obj[0].copy()
        for i in obj[1:]:
            temp += i
        return temp
    else:
        return sums(list(obj))

    return temp


def mult(*obj):
    if type(obj[0]) in [list, tuple]:
        obj = obj[0]
        temp = obj[0]
        for k in obj[1:]:
            temp *= k
        return temp
    else:
        return mult(list(obj))


def sortwithind(a, rev=False):
    # sort a list and record the original index of the sorted elements,
    # return a list which first element is the sorted list and second
    # element is the original index of the elements in the sorted list.
    N = len(a)
    new = [(a[i], i) for i in range(N)]
    asort = sorted(new, key=lambda x: x[0], reverse=rev)
    result = [[x[0] for x in asort], [x[1] for x in asort]]
    return result


def composition(obj, func, dr=0):
    if dr == 0:
        temp = obj[0]
        for k in obj[1:]:
            temp = func(temp, k)
    else:
        temp = obj[-1]
        for k in range(len(obj) - 2, -1, -1):
            temp = func(temp, obj[k])
    return temp


def mean(a):
    return sum(a) / len(a)


def var(a):
    average = mean(a)
    return sum([(x - average)**2 for x in a]) / (len(a) - 1)


def se(a):
    return var(a)**0.5


def norm(a):
    return sum([x**2 for x in a])**0.5


def formatnumber(num):
    if type(num) in [float, fractions.Fraction]:
        y = fractions.Fraction(num).limit_denominator()
        return int(y) if y.__float__().is_integer() else y
    else:
        return num


def formatlist(x):
    result = [formatnumber(i) for i in x]
    return result


def formatrow(x):
    x = [formatlist(i) for i in x]
    return x


def tofloat(x):
    nrow, ncol = x.dim()
    for i in range(nrow):
        for j in range(ncol):
            current = x[i][j]
            if type(current) != int:
                x[i][j] = float(current)


def normal(t):
    x = norm(t)
    return [j / x for j in t]


def sign(a, b):
    if (a > 0 and b > 0) or (a < 0 and b < 0):
        return True
    else:
        return False


def sg(num):
    if num < 0:
        return -1
    elif num > 0:
        return 1
    else:
        return 0


def tofunc(a):
    # transfrom a string in format '[var1, var2, ...] (-> or | or :) [expressions]'
    # to a function lambda [var1, var2, ...] : [expressions]
    for separator in ['=', '->', '|', ':']:
        if separator in a:
            result = a.split(separator)
            return eval(f'lambda {result[0]} : {result[1]}', LOC, GLOB)
    return 'no valid separator was found'


class case:
    '''
    Casepairs are a list with condition and value one by one,
    like condition1, value1, condition2, value2, ...
    The order of pairs is interpreted as if, elif, elif, ...
    default is the value at else case, and this function would
    transfrom casepairs to a case-value list pairs (the reason why I
    not really want to use dictionary here is because the order of
    the cases matters, so an ordered data structure is needed)
    For the else case, you can set the value of default, when
    none of the conditions matches, this function would return default.
    (functions and lambda functions are also fine to be as keys or values)
    
    '''
    def __init__(self,
                 *casepairs,
                 default=None,
                 noelif=False,
                 func=None,
                 toeval=True,
                 toeval2=None,
                 asexp=False,
                 asexp2=False,
                 func_asexp=False):
        if toeval2 is None:
            toeval2 = toeval

        N = len(casepairs)
        if N == 2:
            first = casepairs[0]
            if type(first) in [list, tuple]:
                default = casepairs[1]
                casepairs = first
                N = len(casepairs)
        if N % 2 != 0:
            self.default = casepairs[-1]
            casepairs = casepairs[:-1]
            N -= 1
        else:
            self.default = default
        self.cases = [casepairs[i] for i in range(N) if i % 2 == 0]
        self.values = [casepairs[i] for i in range(N) if i % 2 != 0]
        self.noelif = noelif
        self.func = func
        self.toeval = toeval
        self.toeval2 = toeval2
        if asexp:
            # change all strings in conditions to lambda functions (using lambda + eval)
            for i in range(len(self.cases)):
                self.cases[i] = tofunc(self.cases[i])
        if asexp2:
            # same for values
            for i in range(len(self.values)):
                self.values[i] = tofunc(self.values[i])
        if func_asexp:
            # same for func
            self.func = tofunc(self.func)

    def __repr__(self):
        result = ''
        cases, values = self.cases, self.values
        M = len(cases)
        for i in range(M):
            result += f'case{i+1}:  {cases[i]}\nvalue{i+1}:  {values[i]}\n'
        result += f'default:  {self.default}'
        return result

    def condition(self, n, x=None, toevals=True, ispoly=True):
        try:
            result = eval(n, LOC, GLOB)
            if (result is not True) and (result is not False):
                if toevals:
                    result = (x == result)
                else:
                    result = (x == n)
        except:
            if callable(n):
                if toevals:
                    result = n(*x) if not ispoly else n(x)
                else:
                    result = (x == n)
            else:
                result = (x == n)
        return result

    def value(self, n, x=None, toevals=True, ispoly=True):
        try:
            if toevals:
                result = eval(n, LOC, GLOB)
            else:
                result = n
        except:
            if callable(n):
                result = n(*x) if not ispoly else n(x)
            else:
                if toevals:
                    try:
                        result = exec(n, LOC, GLOB)
                    except:
                        result = n
                else:
                    result = n
        return result

    def __call__(self, *x, toeval=None, toeval2=None):
        ispoly = True
        if len(x) == 1:
            x = x[0]
        else:
            ispoly = False
        if toeval is None:
            toeval = self.toeval
        if toeval2 is None:
            toeval2 = self.toeval2
        cases, values = self.cases, self.values
        if self.func is not None:
            x = self.func(x)
        M = len(cases)
        if not self.noelif:
            for k in range(M):
                fit = self.condition(cases[k],
                                     x,
                                     toevals=toeval,
                                     ispoly=ispoly)
                if fit:
                    return self.value(values[k],
                                      x,
                                      toevals=toeval2,
                                      ispoly=ispoly)

            return self.value(self.default, x, toevals=toeval2, ispoly=ispoly)
        else:
            result = []
            for k in range(M):
                fit = self.condition(cases[k],
                                     x,
                                     toevals=toeval,
                                     ispoly=ispoly)
                if fit:
                    current = self.value(values[k],
                                         x,
                                         toevals=toeval2,
                                         ispoly=ispoly)
                    result.append(current)
            if result == []:
                return self.value(self.default,
                                  x,
                                  toevals=toeval2,
                                  ispoly=ispoly)
            else:
                return result if len(result) > 1 else result[0]

    def append(self, x, y):
        self.cases.append(x)
        self.values.append(y)

    def insert(self, x, y, ind):
        self.cases.insert(ind, x)
        self.values.insert(ind, y)

    def __delitem__(self, ind):
        del self.cases[ind]
        del self.values[ind]

    def __setitem__(self, ind, x):
        self.cases[ind] = x[0]
        self.values[ind] = x[1]

    def __getitem__(self, ind):
        return [self.cases[ind], self.values[ind]]

    def setcase(self, ind, x):
        self.cases[ind] = x

    def setval(self, ind, x):
        self.values[ind] = x

    def getcase(self):
        return self.cases

    def getval(self):
        return self.values

    def getdefault(self):
        return self.default

    def setdefault(self, x):
        self.default = x

    def deldefault(self):
        self.default = None

    def swap(self, a, b):
        self.cases[a], self.cases[b] = self.cases[b], self.cases[a]
        self.values[a], self.values[b] = self.values[b], self.values[a]

    def swapcase(self, a, b):
        self.cases[a], self.cases[b] = self.cases[b], self.cases[a]

    def swapval(self, a, b):
        self.values[a], self.values[b] = self.values[b], self.values[a]

    def reverse(self):
        self.cases.reverse()
        self.values.reverse()

    def reversecase(self):
        self.cases.reverse()

    def reverseval(self):
        self.values.reverse()


def get(a, indlist, func=None):
    if func is None:
        return [a[x] for x in indlist]
    else:
        return [func(a[x]) for x in indlist]


def perform(a, func, indlist=None, args=None):
    if indlist is None:
        if args is None:
            for x in a:
                func(x)
        else:
            for x in a:
                func(x, args)
    else:
        if args is None:
            for x in indlist:
                func(a[x])
        else:
            for x in indlist:
                func(a[x], args)


GT = lambda a, b: a > b
GE = lambda a, b: a >= b
LT = lambda a, b: a < b
LE = lambda a, b: a <= b
EQ = lambda a, b: a == b
RANGE = lambda a, m, n: a in range(m, n)
CONTAIN = lambda a, b: a in b
ADD = lambda a, b: a + b
SUB = lambda a, b: a - b
MULT = lambda a, b: a * b
DIV = lambda a, b: a / b
FDIV = lambda a, b: a // b
MOD = lambda a, b: a % b
NE = lambda a, b: a != b


def AND(*a):
    if isinstance(a[0], list):
        result = AND(*a[0])
    else:
        result = a[0]
    for i in a[1:]:
        if isinstance(i, list):
            result = result and AND(*i)
        else:
            result = result and i
    return result


def OR(*a):
    if isinstance(a[0], list):
        result = OR(*a[0])
    else:
        result = a[0]
    for i in a[1:]:
        if isinstance(i, list):
            result = result or OR(*i)
        else:
            result = result or i
    return result


def NOT(*a):
    a = list(a)
    result = []
    for i in range(len(a)):
        if isinstance(a[i], list):
            result.append(NOT(*a[i]))
        else:
            result.append(not a[i])
    if len(result) == 1:
        return result[0]
    return result


def ifel(exp1, value1, value2):
    return case(exp1, value1, default=value2)


def getbyind(x, ind):
    return x[ind] if isinstance(ind, int) else [x[t] for t in ind]


def pick(elements, picklists, conditions, func=None, target=None):
    N = len(picklists)
    for i in elements:
        found = False
        for j in range(N - 1):
            if conditions[j](i):
                found = True
                (picklists[j].append(i) if target is None\
                 else picklists[j].append(target[i]))\
                    if func is None else\
                    (picklists[j].append(func(i))\
                     if target is None else\
                     picklists[j].append(func(target[i])))

        if not found:
            (picklists[N-1].append(i) if target is None else\
             picklists[N-1].append(target[i])) if func is None\
                else (picklists[N-1].append(func(i)) if target\
                      is None else picklists[N-1].append(func(target[i])))

    return tuple(picklists)


def mass(datatype, num, mode=0):
    if type(datatype) == type:
        return tuple([datatype.__call__() for i in range(num)])
    elif datatype is None:
        return [None for i in range(num)]
    else:
        if isinstance(num, int):
            return {datatype[i] : [datatype[i].__call__()\
            for j in range(num)] for i in\
            range(len(datatype))} if mode == 0 else\
            [[datatype[i].__call__() for j in range(num)]\
            for i in range(len(datatype))]
        else:
            return {datatype[i] : [datatype[i].__call__()\
            for j in range(num[i])] for i in range(len(datatype))}\
            if mode == 0 else [[datatype[i].__call__() for\
            j in range(num[i])] for i in range(len(datatype))]


'''
some examples of mass and pick:

>>> mass(list, 2)
[], []



>>> a, b, c = pick([i for i in range(30)],mass(list,3),[lambda i : i % 3 == 0, lambda i : i % 3 == 1])
>>> a
[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
>>> b
[1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
>>> c
[2, 5, 8, 11, 14, 17, 20, 23, 26, 29]

>>> mass([list,int],[2,3],1)
[[[], []], [0, 0, 0]]
>>> mass([list,int],[2,3])
{<class 'list'>: [[], []], <class 'int'>: [0, 0, 0]}

>>> [a,b],[c,d,e] = mass([list,int],[2,3],1)
>>> a
[]
>>> b
[]
>>> c
0
>>> d
0
>>> e
0
>>> mass([list,int],[1,2],1)
[[[]], [0, 0]]
>>> mass([list,int],[3,2],1)
[[[], [], []], [0, 0]]
>>> [a,b,c],[d,e] = mass([list,int],[3,2],1)
>>> a
[]
>>> b
[]
>>> c
[]
>>> d
0
>>> e
0

'''


class WHILE:
    def __init__(self,
                 varname,
                 start,
                 holdcond,
                 action=None,
                 breakaction=None,
                 change=1,
                 tonext=0):
        # tonext = 0: neseted with the next loop object
        # tonext = 1: parallel with the next loop object
        self.varname, self.start, self.holdcond, self.action,\
            self.breakaction, self.change, self.tonext = varname, start,\
            holdcond, action, breakaction, change, tonext


class FOR:
    def __init__(self,
                 varname,
                 start,
                 end,
                 element=None,
                 action=None,
                 breakaction=None,
                 tonext=0):
        self.varname, self.start, self.end, self.element, self.action,\
            self.breakaction, self.tonext = varname, start, end, element,\
            action, breakaction, tonext


# find matching elements by index and lambda functions or values
#temp = [('a', 1, 1.5), ('b', 2, 5.1), ('c', 9, 4.3)]
find = lambda self, i, value: [
    x for x in self if (callable(value) and value(x[i])) or (x[i] == value)
]
order = lambda a, value, num: [
    x for x in range(len(a))
    if (callable(value) and value(a[x])) or (a[x] == value)
][num]
indorder = lambda a, value, num: [
    a[x] for x in range(len(a))
    if (callable(value) and value(x)) or (x == value)
][num]
ordere = lambda a, value, num: [
    a[x] for x in range(len(a))
    if (callable(value) and value(a[x])) or (a[x] == value)
][num]
#print(find(temp,1,2))
# [('b', 2, 5.1)]
#print(find(temp,1,lambda x: x< 8))
# [('a', 1, 1.5), ('b', 2, 5.1)]


def paraloop(*loop):
    pass


def nestloop(*loop):
    LB = '\n'
    TAB = '\n    '
    space = '    '
    result = ''
    count = 1
    #whindex, findex = pick(list(range(len(loop))), mass(list, 2), [lambda x : isinstance(loop[x], WHILE)])
    N = len(loop)
    for i in range(N):
        current = loop[i]
        if isinstance(current, WHILE):
            result += f'{current.varname} = {current.start}' + LB + (count -
                                                                     1) * space
            result += f'while {current.varname} {current.holdcond} :' + LB + count * space
            if current.action is not None:
                result += current.action + LB + count * space
            if current.breakaction is not None:
                result += f'if {current.breakaction} : break' + LB + count * space
            if current.tonext == 1:
                result += f'{current.varname} += {current.change}' + LB + (
                    count - 1) * space
        elif isinstance(current, FOR):
            if current.element is None:
                result += f'for {current.varname} in range({current.start}, {current.end}) :' + LB + count * space
            else:
                result += f'for {current.varname} in {current.element} :' + LB + count * space
            if current.action is not None:
                result += current.action + LB + count * space
            if current.breakaction is not None:
                result += f'if {current.breakaction} : break' + LB + count * space
            if current.tonext == 1:
                result = result[:-4]
        if i == N - 1:
            if isinstance(current, WHILE):
                result += f'{current.varname} += {current.change}' + LB + (
                    count - 1) * space
            else:
                result = result[:-4]
            temp = count - 1
            temp2 = i
            while temp2 > 0:
                new = loop[temp2 - 1]
                if isinstance(new, WHILE) and new.tonext == 0:
                    result += f'{new.varname} += {new.change}' + LB + (
                        temp - 1) * space
                temp -= 1
                temp2 -= 1
        if current.tonext == 0:
            count += 1

    return result


def redup(a):
    # remove the duplicates of elements in a list while preserving the order.
    return [a[i] for i in range(len(a)) if i == a.index(a[i])]


def lrange(m, n=None, datatype=list):
    if n is None:
        return datatype.__call__([i for i in range(0, m)])
    else:
        return datatype.__call__([i for i in range(m, n)])


def choose(x, conditions, anyis=0, datatype=list):
    if not isls(conditions):
        if not callable(conditions):
            result = [i for i in x if i == conditions]
        else:
            result = [i for i in x if conditions(i)]
    else:
        if anyis == 0:
            result = [i for i in x if all(j(i) for j in conditions)]
        else:
            result = [i for i in x if any(j(i) for j in conditions)]
    return datatype.__call__(result)


chooseif = choose


def swap(x, a, b):
    x[a], x[b] = x[b], x[a]


def expr(expression, separator='/'):
    args, ret = expression.split(separator)
    if ',' not in args:
        args = ', '.join(args.split())
    result = eval(f'lambda {args} : {ret}')
    return result


# make a function to break out the nested loops and the number of loops
# to break out could be specified

#class OuterLoop(Exception):
#pass

#i = 10
#check = True
#try:
#while True:
#while check:
#i -= 1
#if i < 0:
#raise OuterLoop
#except OuterLoop:
#print('success')


def irange(a, b=None, step=1):
    if b is None:
        return irange(0, a, step)
    return range(a, b + 1, step)


def rrange(a, b=None, step=1):
    if b is None:
        return irange(1, a, step)
    return range(a, b + 1, step)


class timer:
    def __init__(self, digit=3):
        self.begins, self.stops, self.begin_time, self.stop_time = mass(
            None, 4)
        self.digit = digit
        self.has_start = False

    def start(self, show=True, current=True):
        if self.has_start:
            return 'the timer has already started'
        self.begins = time.time()
        self.begin_time = str(datetime.datetime.now())
        self.has_start = True
        if show:
            print('the timer starts now')
        if current:
            print(f'current time when it starts is {self.begin_time}')

    def stop(self, duration=True, current=True, show=True):
        if not self.has_start:
            return 'the timer has not yet started'
        self.stops = time.time()
        self.stop_time = str(datetime.datetime.now())
        self.has_start = False
        spend = self.stops - self.begins
        second = rounding(spend % 60, self.digit)
        minute = int(spend // 60)
        hour = int(spend // 3600)
        self.duration = f'{hour}h {minute}m {second}s'
        if show:
            print('the timer stops now')
        if duration:
            print(f'the time pasts {self.duration}')
        if current:
            print(f'current time when it stops is {self.stop_time}')

    def last_record(self):
        print(f'last record time pass is {self.duration}')

    def precision(self, digit):
        self.digit = digit

    def get_precision(self):
        return self.digit

    def last_begin(self):
        return self.begin_time if self.begin_time is not None else time.ctime(
            self.begins)

    def last_stop(self):
        return self.stop_time if self.stop_time is not None else time.ctime(
            self.stops)

    def last(self):
        return self.last_begin(), self.last_stop()

    def counts(self, unit=1, length=60, digit=3, refresh=False):
        if self.has_start:
            return 'the timer has already started'
        self.has_start = True

        if not refresh:
            begins = time.time()
            self.begins = begins
            self.begin_time = str(datetime.datetime.now())
            past = time.time() - begins
            print(f'time past: {0:.{digit}f} s')
            while past < length:
                time.sleep(unit)
                past = rounding(time.time() - begins, digit)
                print(f'time past: {past} s')
        else:
            begins = time.time()
            self.begins = begins
            self.begin_time = str(datetime.datetime.now())
            past = time.time() - begins
            print(f'time past: {0:.{digit}f} s')
            while past < length:
                time.sleep(unit)
                past = rounding(time.time() - begins, digit)
                os.system('cls')
                print(f'time past: {past} s')

        self.stops = time.time()
        self.stop_time = str(datetime.datetime.now())
        self.has_start = False
        print(f'time count completes, length: {length} s, interval: {unit} s')
        input('press enter to continue')


# a = timer()
# a.counts(0.01,20,3, refresh = True)
class exp:
    # a class stores the information of an expression and has many functionalities
    # to associate with other expressions.
    # an exp instance could be compiled into an AST object to analyze.
    def __init__(self,
                 expstr,
                 exptype=None,
                 precedence='right',
                 globs=GLOB,
                 locs=LOC):
        self.expstr = expstr
        self.exptype = exptype
        self.precedence = precedence
        self.globs = globs
        self.locs = locs

    def __call__(self, other=None):
        if other is None:
            return eval(self.expstr, self.globs, self.locs)

    def __repr__(self):
        return self.expstr

    def __add__(self, other):
        # concatenate this expression with another expression
        return exp(f'{self.expstr}, {other.expstr}', self.exptype,\
                   self.precedence, self.globs, self.locs)

    def cat(self, other, mode='simple'):
        if mode == 'simple':
            return self + other
        else:
            return exp(f'{self.expstr} {mode} {other.expstr}', self.exptype,\
                       self.precedence, self.globs, self.locs)


class SyntaxTree:
    # store the information of syntax you define, and are able to return
    # a translator that follows the rules in this syntax tree.
    # this translator could translate your own scripts written in your
    # own defined language to be runnable
    pass


class Translator:
    def __init__(self, operators=[]):
        self.operators = operators
        if self.operators != []:
            self.prefix, self.infix, self.suffix =\
            pick(self.operators, mass(list,3),\
            [lambda x: x.place == PRE, lambda x : x.place == INF])
        else:
            self.prefix, self.infix, self.suffix = mass(list, 3)

    def opesort(self):
        operators = self.operators
        head = choose(operators, lambda x: x.lowerthan == [])
        head += merge([[i for i in x.same if i not in head] for x in head])
        bottom = [x for x in operators if x.higherthan == [] and x not in head]
        bottom += merge([[i for i in x.same if i not in bottom]
                         for x in bottom])
        whole = []
        middle = []
        temp = head[:]
        while True:
            if len(temp) == 0:
                break
            else:
                temp = merge([[j for j in x.higherthan if j not in bottom]
                              for x in temp if x.higherthan != []])
                if temp != []:
                    middle.append(temp)
        if head != []:
            whole.append(head)
        if middle != []:
            whole.append(middle)
        if bottom != []:
            whole.append(bottom)
        return whole

    def level(self, i):
        return self.opesort[i]

    def pres(self):
        return self.prefix

    def infs(self):
        return self.infix

    def sufs(self):
        return self.suffix

    def opes(self):
        return self.operators

    def trans(self,
              y,
              var=None,
              val=None,
              var2=None,
              val2=None,
              notresult=False):
        if var is not None:
            if y.place == INF:
                y.bindvar(var, val, var2, val2)
            else:
                y.bindvar(var, val)
        if notresult:
            return str(y)
        else:
            return y.play()

        return y.play

    def readline(self, unformatedline):
        whole = self.opesort()
        found = False
        result = unformatedline
        for opeset in whole:
            for eachope in opeset:
                opestr = str(eachope)
                if opestr in unformatedline:
                    found = True
                    length = len(eachope)
                    ind = unformatedline.index(opestr)
                    if eachope.place == PRE:
                        val = unformatedline[ind + length:]
                        val = ''.join(
                            [i for i in val if i.isdigit() or i.isalpha()])
                        result = eachope.func(eval(val), LOC, GLOB)
                    elif eachope.place == INF:
                        val1 = unformatedline[:ind]
                        val2 = unformatedline[ind + length:]
                        val1 = ''.join(
                            [i for i in val1 if i.isdigit() or i.isalpha()])
                        val2 = ''.join(
                            [i for i in val2 if i.isdigit() or i.isalpha()])
                        result = eachope.func(eval(val1, LOC, GLOB),
                                              eval(val2, LOC, GLOB))
                    elif eachope.place == SUF:
                        val = unformatedline[:ind]
                        val = ''.join(
                            [i for i in val if i.isdigit() or i.isalpha()])
                        result = eachope.func(eval(val, LOC, GLOB))
                    break
            if found:
                break
        return result

    def update(self, x):
        if not isls(x):
            self.operators.append(x)
            self.prefix.append(x) if x.place == PRE else\
            (self.infix.append(x) if x.place == INF else self.suffix.append(x))
        else:
            for i in x:
                self.update(i)


def sumup(*x):
    if len(x) == 1 and isinstance(x[0], list):
        x = x[0]
        result = x[0]
        for i in x[1:]:
            result += i
        return result
    else:
        return sumup(list(x))


merge = lambda x: [i for t in x
                   for i in merge(t)] if isinstance(x, list) else [x]

PRE = 'prefix'
SUF = 'suffix'
INF = 'infix'


class Ope:
    def __init__(self,
                 pattern,
                 place,
                 func=None,
                 lowerthan=[],
                 higherthan=[],
                 same=[],
                 special=[]):
        # lowerthan: a list/set/tuple of all the operators that self operator has
        # lower precedence comparing to.
        # higherthan: a list/set/tuple of all the operators that self operator has
        # higher precedence comparing to.
        # same: a list/set/tuple of all the operators that self operator has
        # the same precedence comparing to. When two operators have the
        # same precedence, it means the calculations would be from left
        # to right, one by one, and the calculation order is naturally straighforward.
        # If an operator A is lower than another operator B, all of the operators
        # that is lower than operator A are also lower than the operator B,
        # same thing for higherthan cases. However, there would be some special
        # cases in some situations, and you are able to define them through
        # modifying the variable special.

        self.pattern, self.place, self.func, self.higherthan,\
        self.lowerthan, self.same, self.special = pattern, place,\
        func, higherthan[:], lowerthan[:], same[:], special[:]
        for i in self.higherthan:
            if self not in i.lowerthan:
                i.lowerthan.append(self)
        for j in self.lowerthan:
            if self not in j.higherthan:
                j.higherthan.append(self)
        for k in self.same:
            if self not in k.same:
                k.same.append(self)
        self.var = None
        self.varval = None

    def __len__(self):
        return len(str(self))

    def __repr__(self):
        if self.var is None:
            return self.pattern
        else:
            if self.place == INF:
                return f'{self.var[0]} {self.pattern} {self.var[1]}'
            if self.place == PRE:
                return f'{self.pattern}{self.var}'
            elif self.place == SUF:
                return f'{self.var}{self.pattern}'

    def __le__(self, other):
        return self in other.higherthan

    def __gt__(self, other):
        return self in other.lowerthan

    def __eq__(self, other):
        if type(other) != Ope:
            return False
        return self.pattern == other.pattern and self.place == other.place\
        and self.func == other.func

    def copy(self):
        return cp(self)

    def play(self):
        if self.func is None:
            raise ValueError('this operator does not have any functionalities')
        if self.var is None or self.varval is None:
            raise ValueError('lack of variables or values of variables')
        varval = self.varval
        if self.place == INF:
            return f'{self} = {self(varval[0], varval[1])}'
        return f'{self} = {self(varval)}'

    def setfun(self, func):
        self.func = func

    def setpattern(self, x):
        self.pattern = x

    def setplace(self, t):
        self.place = t

    def addlt(self, x):
        if x not in self.lowerthan:
            self.lowerthan.append(x)
        if self not in x.higherthan:
            x.higherthan.append(self)

    def addht(self, x):
        if x not in self.higherthan:
            self.higherthan.append(x)
        if self not in x.lowerthan:
            x.lowerthan.append(self)

    def addse(self, x):
        if x not in self.same:
            self.same.append(x)
        if self not in x.same:
            x.same.append(self)

    def addsp(self, x):
        self.special.append(x)

    def ht(self):
        '''
        >>> a = Ope('$',PRE)
        >>> b = Ope('_+_',INF)
        >>> a.ht()
        []
        >>> a.higherthan = [b]
        >>> a.ht()
        [ _+_ ]
        >>> b.higherthan = [Ope('@^@',INF)]
        >>> b
         _+_ 
        >>> a.ht()
        [ _+_ ,  @^@ ]

        '''
        return self.higherthan + merge(
            [i.ht() for i in self.higherthan if i.higherthan != []])

    def lt(self):
        return self.lowerthan + merge(
            [i.lt() for i in self.lowerthan if i.lowerthan != []])

    def setvar(self, varset1, varset2=None):
        if self.place == INF:
            if varset2 is None:
                raise ValueError(
                    'Infix operators require two variables to be set up.')
            self.var = [varset1[0], varset2[0]]
            self.varval = [varset1[1], varset2[1]]
        else:
            if varset2 is not None:
                raise ValueError(
                    'too much variables for an unary operator to bind with')
            self.var = varset1[0]
            self.varval = varset1[1]

    def __call__(self, a=None, b=None):
        if self.place == INF:
            return self.func(a, b)
        else:
            return self.func(a)


class Multope(Ope):
    # An operator type for those operators that have 3 or
    # more than 3 variables needed to be operated.
    # For example, an ternary operator in C: a = (condition) ? b : c
    # means a = b if (condition) is True, else a = c,
    # in python, such thing is written like: a = b if (condition) else c
    # here "? :" is an ternary operator, and could be more proper to be called
    # as a ternary operator set rather than a ternary operator. This set
    # has two operators, neither is unary or binary, but could be put together
    # to make a great functionality.
    def __init__(self, pattern=[], place=3, func=None):
        super().__init__(pattern, place, func)


def fac(n):
    if n == 0:
        return 1
    result = n
    for i in range(1, n):
        result *= i
    return result


def P(n, r):
    return fac(n) // fac(n - r)


def C(n, r):
    return fac(n) // (fac(r) * fac(n - r))


def perm(n, k=None):
    # return all of the permutations of the elements in x
    if isinstance(n, int):
        n = list(range(1, n + 1))
    if isinstance(n, str):
        n = list(n)
    if k is None:
        k = len(n)
    return eval(
        f'''[{f"[{', '.join([f'n[a{i}]' for i in range(k)])}]"} {''.join([f'for a{i} in range(len(n)) ' if i == 0 else f"for a{i} in range(len(n)) if a{i} not in [{', '.join([f'a{t}' for t in range(i)])}] " for i in range(k)])}]''',
        locals())


def comb(n, k=None):
    # return all of the combinations of the elements in x
    if isinstance(n, int):
        n = list(range(1, n + 1))
    if isinstance(n, str):
        n = list(n)
    if k is None:
        k = len(n)
    return eval(
        f'''[{f"[{', '.join([f'n[a{i}]' for i in range(k)])}]"} {''.join([f'for a{i} in range(len(n)-k+1) ' if i == 0 else f'for a{i} in range(a{i-1} + 1, {i} + len(n)-k+1) ' for i in range(k)])}]''',
        locals())


def allcomb(n):
    if isinstance(n, int):
        n = list(range(1, n + 1))
    if isinstance(n, str):
        n = list(n)
    return sumup([comb(n, i) for i in range(1, len(n) + 1)])


def allperm(n):
    if isinstance(n, int):
        n = list(range(1, n + 1))
    if isinstance(n, str):
        n = list(n)
    return sumup([perm(n, i) for i in range(1, len(n) + 1)])


def sets(ind, c, x, mode=0):
    if mode == 1:
        b = list(x)
        sets(ind, c, b, 2)
        x = ''.join(b)
        return x
    else:
        if mode == 2:
            if not isinstance(ind, list):
                ind = [ind]
        if mode == 0:
            if not isinstance(ind[0], list):
                ind = [ind]
        if type(c) not in [list, tuple, set]:
            for i in ind:
                x[i] = c
        else:
            N = min(len(ind), len(c))
            for i in range(N):
                x[ind[i]] = c[i]


def index(a, value, element=False):
    if callable(value):
        if not element:
            return next(i for i, j in enumerate(a) if value(j))
        return next(i for i in a if value(i))
    return a.index(value)


def dowhile(conditions, actions, start=0, changes=1, usei=False, output=False):
    if type(conditions) != list:
        conditions = [conditions]
    if type(actions) != list:
        actions = [actions]
    i = start
    conditions = [(x, ) if type(x) != tuple else x for x in conditions]
    actions = [(x, ) if type(x) != tuple else x for x in actions]
    while all((x[0](x[1]) if not usei else x[0](i)) for x in conditions):
        for t in range(len(actions)):
            current = actions[t]
            if len(current) > 1:
                actions[t] = tuple([current[0], current[0](current[1])])
            else:
                current[0](i)
        i += changes
    if output:
        result = [x[1] for x in actions if isinstance(x, tuple)]
        if len(result) == 1:
            result = result[0]
        return result


def dofor(obj, actions, getobj=False, getother=False):
    if type(actions) != list:
        actions = [actions]
    new = []
    other = []
    if isinstance(obj, range):
        obj = list(obj)
    for i in obj:
        for t in range(len(actions)):
            current = actions[t]
            if isinstance(current, tuple):
                actions[t] = tuple([current[0], current[0](current[1])])
            elif callable(current):
                new.append(current(i))
    if getobj:
        return new
    if getother:
        result = [x[1] for x in actions if isinstance(x, tuple)]
        if len(result) == 1:
            result = result[0]
        return result


class gt(Ope):
    def __init__(self):
        super().__init__('>', INF, lambda a, b: a > b)


class le(Ope):
    def __init__(self):
        super().__init__('<', INF, lambda a, b: a < b)


class eq(Ope):
    def __init__(self):
        super().__init__('==', INF, lambda a, b: a == b)


class Not(Ope):
    def __init__(self):
        super().__init__('!', PRE, lambda a: not a)


class Fac(Ope):
    def __init__(self):
        super().__init__('!', SUF, fac)


def cby(obj, func, dr=0):
    if dr == 0:
        temp = obj[0]
        for k in obj[1:]:
            temp = func(temp, k)
    else:
        temp = obj[-1]
        for k in range(len(obj) - 2, -1, -1):
            temp = func(temp, obj[k])
    return temp


def isls(a):
    return type(a) in [list, tuple, set]


def polyls(a, standard=None):
    if standard is None:
        return a if isls(a) else [a]
    return a + [a[-1] for i in range(standard - len(a))] if isls(a) else [
        a for i in range(standard)
    ]


def dolist(a, b, func):
    N = min(len(a), len(b))
    return [func(a[i], b[i]) for i in range(N)]


def donlist(*alist, func=lambda x, y: x + y, inner=False):
    if len(alist) > 1:
        alist = list(alist)
    else:
        alist = alist[0]
    result = alist[0]
    if not inner:
        for i in alist[1:]:
            result = func(result, i)
    else:
        for i in alist[1:]:
            result = dolist(result, i, func)
    return result


def doto(*alist, func, new=False, copy=False):
    if len(alist) > 1:
        return doto(list(alist), func=func, new=new)
    else:
        alist = alist[0]
        if copy:
            alist = cp(alist)
        result = [func(x) for x in alist]
        return result if not new else alist


def doton(alist, func, new=False, copy=False):
    # do something as you like to n lists
    if copy:
        alist = cp(alist)
    result = [func(x) for x in alist]
    return result if not new else alist


def funcls(*func, way='and', ind=None):
    if len(func) > 1:
        return funcls(list(func), way=way)
    else:
        func = func[0]
        if ind is not None:
            func = [func[x] for x in ind]
        if way == 'and':
            return lambda x: all(i(x) for i in func)
        elif way == 'or':
            return lambda x: any(i(x) for i in func)
        elif way == 'do':

            def funcs(x, copy=False):
                if copy:
                    x = cp(x)
                for i in func:
                    i(x)
                return x

            return funcs


def cb(a, b):
    # combine a and b with type check
    types = type(a)
    if types == list:
        return a + [b]
    elif types == tuple:
        return a + (b, )
    elif types == set:
        return a | {b}
    elif types == str:
        return a + b
    else:
        return [a] + [b]


def countif(a, func, anyis=0, show=0, datatype=list):
    if not isls(func):
        if not callable(func):
            return a.count(func)
        else:
            result = [i for i in a if func(i)]
            return datatype.__call__(result) if show != 0 else len(result)
    else:
        if anyis == 0:
            result = [i for i in a if all(j(i) for j in func)]
        else:
            result = [i for i in a if any(j(i) for j in func)]
        return datatype.__call__(result) if show != 0 else len(result)


def indif(a, func, anyis=0, datatype=list):
    if not isls(func):
        if not callable(func):
            result = [i for i in range(len(a)) if a[i] == func]
        else:
            result = [i for i in range(len(a)) if func(a[i])]
    else:
        if anyis == 0:
            result = [i for i in range(len(a)) if all(j(a[i]) for j in func)]
        else:
            result = [i for i in range(len(a)) if any(j(a[i]) for j in func)]
    return datatype.__call__(result)


def mixrange(alist, types=list, interlace=False, surplus=False):
    # alist could be a list of ranges, lists, tuples, sets, or any iterable objects
    # when interlace is set to True, the range will be combined as each from every list every time(interlaced)
    # when surplus is set to True, if there are any remaining parts of the ranges after interlacing,
    # add them to the bottom of the mixed ranges.
    result = []
    if not interlace:
        for i in alist:
            result += list(i)
    else:
        N = min([len(i) for i in alist])
        for k in range(N):
            for j in alist:
                result.append(j[k])
        if surplus:
            for t in alist:
                M = len(t)
                for r in range(N, M):
                    result.append(t[r])
    return result if types == list else types.__call__(result)