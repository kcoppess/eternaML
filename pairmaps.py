# data processing
import numpy as np

f = open('structures.txt','r')
struct = []
maxLength = 0
b = f.readline()
while len(b) > 1:
    if len(b)-1 > maxLength:
        maxLength = len(b) - 1
    struct.append(b)
    b = f.readline()
f.close()

pairmaps = []
pairmaps1D = []
loops = []

size = 640
for b in struct:
    n = len(b)-1 # ignoring endline character
    pm = np.zeros((size,size))
    pm1d = np.zeros(size)
    l = np.zeros(size)
    for i in range(n):
        pm1d[i] = -1
        pm[i,i] = -1
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
    pm = np.concatenate(pm)
    pairmaps.append(pm) # flattens matrix
    pairmaps1D.append(pm1d)
    loops.append(l)
