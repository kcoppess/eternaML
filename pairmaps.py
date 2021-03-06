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
num_zeros = 0
num_ones = 0
num_twos = 0
num_threes = 0
maximum = len(struct[0])
for b in struct:
    n = len(b)-1 # ignoring endline character
    for m in range(n): # labeling the mth base
        pm = np.zeros((size,size))
        pm1d = np.zeros(size)
        l = np.zeros(6)
        # prepare pairmaps
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
        if b[m] == '(':
            nBranch = 1
            count = 1
            subcount = 0
            for j in range(m+1, n):
                if b[j] == '(':
                    if subcount == 0:
                        nBranch += 1
                    subcount += 1
                    count += 1
                elif b[j] == ')':
                    subcount += -1
                    count += -1
                if count == 0 and pm1d[i] != pm1d[i+1] + 1:
                    l[nBranch] = 1
                    break
            if nBranch == 1:
                if num_ones%50 == 0:
                    pm = np.concatenate(pm)
                    pairmaps.append(pm) # flattens matrix
                    pairmaps1D.append(pm1d)
                    loops.append(l)
                num_ones += 1
            elif nBranch == 2:
                if num_twos%100 == 0:
                    pm = np.concatenate(pm)
                    pairmaps.append(pm) # flattens matrix
                    pairmaps1D.append(pm1d)
                    loops.append(l)
                num_twos += 1
            elif nBranch == 3:
                if num_threes%1 == 0:
                    pm = np.concatenate(pm)
                    pairmaps.append(pm) # flattens matrix
                    pairmaps1D.append(pm1d)
                    loops.append(l)
                num_threes += 1
            else:
                pm = np.concatenate(pm)
                pairmaps.append(pm) # flattens matrix
                pairmaps1D.append(pm1d)
                loops.append(l)
        else:
            if num_zeros%200 == 0:
                l[0] = 1
                pm = np.concatenate(pm)
                pairmaps.append(pm) # flattens matrix
                pairmaps1D.append(pm1d)
                loops.append(l)
            num_zeros += 1
np.savetxt('pairmap2D.csv',pairmaps,delimiter=',')
np.savetxt('loops.csv', loops, delimiter=',')
