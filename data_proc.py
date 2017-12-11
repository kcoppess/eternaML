import pandas as pd
import numpy as np

experts = []
expert_uid = pd.read_csv('experts.txt', delimiter=',', usecols=[0], parse_dates=['uid'])
#print expert_uid.loc[0, 'uid']
for i in range(14):
    experts.append(expert_uid.loc[i,'uid'])
#print experts

target_structures = pd.read_csv('puzzle-structure-data.txt', delimiter='\t', usecols=[0,1], parse_dates=['pid','structure'])
targets = target_structures.set_index('pid')
#print targets.loc['66','structure']
#print targets.iloc[0,1]

#x = '6502997'
playouts = pd.read_csv('moveset6-22a.txt', delimiter='\t', usecols=[1,2,3], nrows=1, parse_dates=['pid', 'uid', 'move_set'])
#playouts = playouts_readin.set_index(['pid','uid'])
#y = eval(playouts.loc[0,'move_set'])
#print y['begin_from']
#y = eval(x.at[0,'move_set'])

#print y['begin_from']

def base_id(b):
    if b == 'A':
        return 1
    elif b == 'U':
        return 2
    elif b == 'G':
        return 3
    elif b == 'C':
        return 4
    else:
        print "invalid base"
        return 0

def sequence_id(seq):
    num_seq = np.zeros(len(seq))
    for i in range(len(seq)):
        num_seq[i] = base_id(seq[i])
    return num_seq

def pairmaps(struc):
    pm1 = np.zeros(size)
    pm2 = np.zeros((size,size))
    n = len(struc)
    for k in range(n):
        pm2[k,k] = -1
        if struc[k] == '(':
            count = 1
            for j in range(k+1, n):
                if struc[j] == '(':
                    count += 1
                elif struc[j] == ')':
                    count += -1
                if count == 0:
                    pm1[k] = j
                    pm1[j] = k
                    pm2[k,j] = 1
                    pm2[j,k] = 1
                    break
        else:
            if pm1[k] == 0 and pm1[0] != k:
                pm1[k] = -1
    return pm1, pm2

N = 2

size = 80

onedim_pairmaps = []
twodim_pairmaps = []
onedim_sequences = []
onedim_locations = []
bases = []
'''
pid = playouts.loc[0,'pid']
for l in range(20674):
    if targets.iloc[l,0] == pid:
        target_struc = targets.iloc[l,1]
        onedim_target, twodim_target = pairmaps(target_struc)
        print onedim_target
        print twodim_target
        break
'''
c = 0
i = 0
while i < N and c < 429762:
    pid = playouts.loc[i]['pid'] #iloc[i,0]
    pid_i = pd.Index([pid])
    for l in range(20674):
        try targets.loc[pid_i]:
            target_struc = targets.iloc[l,1]
            onedim_target, twodim_target = pairmaps(target_struc)
            print onedim_target
            print twodim_target
            break
    uid = playouts.loc[i,'uid']
    moveset = eval(playouts.loc[i,'move_set'])
    moves = moveset['moves']
    orig_sequence = sequence_id(moveset['begin_from'])
    current_sequence = orig_sequence
    for m in moves:
        for j in range(len(m)):
            loc = np.zeros(len(current_sequence))
            base = np.zeros(4)
            pos = m[j]['pos'] - 1
            b = base_id(m[j]['base'])
            
            loc[pos] = 1
            base[b - 1] = 1

            onedim_sequences.append(current_sequence)
            onedim_locations.append(loc)
            bases.append(base)
            onedim_pairmaps.append(onedim_target)
            twodim_pairmaps.append(np.concatenate(twodim_target))
            current_sequence[pos] = b
    i +=1
    c += 1
odp = np.asarray(onedim_pairmaps)
tdp = np.asarray(twodim_pairmaps)
seq = np.asarray(onedim_sequences)
locations = np.asarray(onedim_locations)
b = np.asarray(bases)
print np.shape(odp)
#np.savetxt('onedim_pairmaps.csv', odp, delimiter=',')
#np.savetxt('twodim_pairmaps.csv', tdp, delimiter=',')
#np.savetxt('onedim_sequences.csv', seq, delimiter=',')
#np.savetxt('onedim_locations.csv', locations, delimiter=',')
#np.savetxt('bases.csv', b, delimiter=',')

