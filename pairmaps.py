import numpy as np

f = open('structures.txt','r')

pairmaps = []
pairmaps1D = []
loops = []

b = f.readline()
while len(b) > 1:
    n = len(b)-1 # ignoring endline character
    pm = np.zeros((n,n))-np.identity(n)
    pm1d = -np.ones(n)
    l = np.zeros(n)
    for i in range(n):
        if b[i] == '(':
            count = 1
            for j in range(i+1, n):
                if b[j] == '(':
                    count += 1
                elif b[j] == ')':
                    count += -1
                if count == 0:
                    pm[i,j] = 1
                    pm[j,i] = 1
                    pm1d[i] = j
                    pm1d[j] = i
                    break
    for i in range(n):
        if b[i] == '(':
            nBranch = 1
            count = 1
            subcount = 0
            for j in range(i+1, n):
                if b[j] == '(':
                    if subcount == 0:
                        nBranch += 1
                    subcount += 1
                    count += 1
                elif b[j] == ')':
                    subcount += -1
                    count += -1
                if count == 0 and pm1d[i] != pm1d[i+1] + 1:
                    l[i] = nBranch
                    break
    pairmaps.append(pm)
    pairmaps1D.append(pm1d)
    loops.append(l)
    b = f.readline()
f.close()

