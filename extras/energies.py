# independent nearest-neighbor energy functions for different
# types of secondary structure motifs
# ref: Mathews et al, J Mol Bio 1999
#      Xia et al, Biochemistry 1998
import numpy as np

R = 0.0019872 # kcal/K/mol universal gas constant
T = 298.15 # K temperature (room temp)
RT = R*T

initEnerHairpin = [5.7, 5.6, 5.6, 5.4, 5.9, 5.6, 6.4] # kcal/mol; for numNucleotides = 3 to 9

# bp1+bp2 (starting from terminal end)
nearestNeighbor = { 'AUAU' : -0.93, 'UAUA' : -0.93, 'AUUA' : -1.10, 'UAAU' : -1.33, 
        'CGUA' : -2.08, 'AUGC' : -2.08, 'CGAU' : -2.11, 'UAGC' : -2.11, 
        'GCUA' : -2.24, 'AUCG' : -2.24, 'GCAU' : -2.35, 'UACG' : -2.35,
        'CGGC' : -2.36, 'GCGC' : -3.26, 'CGCG' : -3.26, 'GCCG' : -3.42,
        'AUGU' : -0.55, 'UGUA' : -0.55, 'AUUG' : -1.36, 'GUUA' : -1.36,
        'CGGU' : -1.41, 'UGGC' : -1.41, 'CGUG' : -2.11, 'GUGC' : -2.11,
        'GCGU' : -1.53, 'UGCG' : -1.53, 'GCUG' : -2.51, 'GUCG' : -2.51,
        'GUAU' : -1.27, 'UAUG' : -1.27, 'GUGU' : -0.50, 'UGUG' : -0.50,
        'GUUG' : +1.29, 'AUGU' : -1.00, 'UGGU' : +0.30 }
# NOTE MISSING GU ENVIRONMENT BONUS both in stacked pairs and for hairpin loop closure

terminal = { 'GU' : 0.45, 'UG' : 0.45, 'AU' : 0.45 , 'UA' : 0.45, 'GC' : 0.00 , 'CG' : 0.00 }

terminalMismatch = { 'GA' : 1.0 }

initiation = 4.09

# NOTE numNucleotides >= 3
def hairpin(numNucleotides, closingPair, mismatchPair, specialGU, oligoC):
    free_energy = terminal[closingPair]
    if numNucleotides < 9: # initiation
        free_energy += initEnerHairpin[ numNucleotides - 3 ]
    else:
        free_energy += initEnerHairpin[-1] + 1.75*RT*np.log(numNucleotides / 9.)
    if numNucleotides > 4: # mismatch stacking
        free_energy += terminalMismatch[mismatchPair]
        if mismatchPair == 'UU' or mismatchPair == 'GA': # special bonus
            free_energy += -0.8
    if specialGU: # bonus for condition on GU closing pair
        free_energy += -2.2
    if oligoC: # penalty for loop only having C
        if numNucleotides == 3:
            free_energy += 1.4
        else:
            free_energy += 0.3 * numNucleotides + 1.6
    print free_energy
    return

hairpin(10, 'GC', 'GA', True, False)
