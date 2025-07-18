from src.mathflow import Expression

f = Expression('sqrt(x)')
f.pade(2, 2, 1, backend='verbose')
