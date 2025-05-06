import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(index):

    y_axis1 = pd.read_csv('datasets/val_features_v1.csv').values.astype(np.float32)
    y_axis2 = pd.read_csv('datasets/val_features_v5.csv').values.astype(np.float32)
    y_axis3 = pd.read_csv('datasets/val_features_v10.csv').values.astype(np.float32)

    y_axis1 = y_axis1[index]
    y_axis2 = y_axis2[index]
    y_axis3 = y_axis3[index]

    x_axis = np.empty(102)
    x_axis[1] = 0.05
    x_axis[2] = 0.1
    for x in range(99):
        x_axis[x+3] = x_axis[x+2]+0.1
    
    plt.figure(figsize=(9, 5))

    plt.plot(x_axis, y_axis1)
    plt.title('Simulator Output (CG1_v1)')
    plt.xlabel('Drain Voltage (V)')
    plt.ylabel('Drain Current (A)') 
    plt.show()

    plt.figure(figsize=(9, 5))

    plt.plot(x_axis, y_axis2)
    plt.title('Simulator Output (CG1_v5)')
    plt.xlabel('Drain Voltage (V)')
    plt.ylabel('Drain Current (A)') 
    plt.show()

    plt.figure(figsize=(9, 5))

    plt.plot(x_axis, y_axis3)
    plt.title('Simulator Output (CG1_v10)')
    plt.xlabel('Drain Voltage (V)')
    plt.ylabel('Drain Current (A)') 
    plt.show()