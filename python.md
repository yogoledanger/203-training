# Basics

## Tools
Anaconda, python version 3, IPython, Spyder...

## Zen of Python

```python
import this
```

## Data Types and Containers
int, float, strings, boolean
tuples, lists, dicts, sets
```python
i = 1
price = 1435.9
name = 'Hello'
is_ok = True

1 == True
0 == True
a is None

t = (1, 2, 3)
t2 = (1, 235.45, 'hello', True)
print(t[1])

l = [1,2,3]
l.append(4)
l.extend([5,6])

l[1]
l[1:3]
l[1:]
l[-1]
l[:-1]
len(l)

d = {'k1': 'A', 'k2': 'B'}
d['k3'] = 'C'

d.keys()
d.items()
d.values()
d['k3']

s = {1, 2, 3}
s = set([1, 2, 2, 3])
```
variable as reference

mutable vs immutable

# Control Structures

```python
for i in l:
    if i % 2:
        print(i)

#modulo:
9 % 2 == 1
#floor division
9 // 2 == 4

for i in range(10):
    print(i) 

i = 0
while i < 10:
    print(i)
    i += 1
```


## functions

```python
def fibo(n):
    output = []
    a, b = 0, 1
    for i in range(n):
        print('a is replaced by b')
        print('b is replaced by a+b')
        output.append(a)
    return output
```



list comprehension
```python
[i % 2 for i in range(10)]

[i for i in range(10) if not i % 2]
```

# Exercises

## Pricing of a european call option with the Black Scholes formula
Formula available here: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model

```python

from math import log, sqrt, exp
from scipy import stats

# stats.norm.cdf --> cumulative distribution function
# for normal distribution

def bsm_call_value(S0, K, T, r, sigma):
''' Valuation of European call option in BSM model.
Analytical formula.
Parameters
==========
S0 : float
initial stock/index level
K : float
strike price
T : float
maturity date (in year fractions)
r : float
constant risk-free short rate
sigma : float
volatility factor in diffusion term
Returns
=======
value : float
present value of the European call option
'''
    return 0

c = bsm_call_value(100, 110, 2, 0.02, 0.1)
assert round(c, 3)== 3.391
```

## Functional programming

## Generators

## Object oriented programming

