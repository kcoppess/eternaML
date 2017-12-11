import numpy as np

pairmaps = np.loadtxt('single_playout_pid6502997/onedim_pairmaps.csv', delimiter=',')
sequences = np.loadtxt('single_playout_pid6502997/onedim_sequences.csv', delimiter=',')
locations = np.loadtxt('single_playout_pid6502997/onedim_locations.csv', delimiter=',')
twodim_pairmaps = np.loadtxt('single_playout_pid6502997/twodim_pairmaps.csv', delimiter=',')

size = 80

sequences2 = np.tile(sequences, (1,size))

loc_features = np.concatenate((twodim_pairmaps, sequences2), axis = 1)
np.savetxt('single_playout_pid6502997/twolocation_features.csv', loc_features, delimiter = ',')

locations2 = np.tile(locations, (1,size))

base_features = np.concatenate((loc_features, locations2), axis = 1)
np.savetxt('single_playout_pid6502997/twobase_features.csv', base_features, delimiter = ',')

