import matplotlib.pyplot as plt

step = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
accuracy = [0.014809, 0.0622569, 0.097309, 0.181753, 0.39099, 0.942986, 0.988455, 0.988507, 0.988507, 0.988507, 0.988507, 0.988507, 0.988507]
simpler = [0.0425, 0.146215, 0.244809, 0.357049, 0.563819, 0.854462, 0.988333, 0.988507, 0.988507, 0.988507, 0.988507, 0.988507, 0.988507]
plt.plot(step, accuracy, linewidth = 2)
plt.plot(step, simpler, linewidth = 2, label='Fewer features')
plt.legend()
plt.xlabel('Training Iterations')
plt.ylabel('Training Accuracy')
plt.show()
