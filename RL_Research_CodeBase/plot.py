import matplotlib.pyplot as plt
import scipy.io
import math

# Distance of RCSD from BS and PDs
distance=[math.sqrt(1+x**2) for x in range(1,101,2)]
distance2=[math.sqrt(1+x**2) for x in range(1,102,25)]

# Transmit power of primary devices
power=[0.5+0.175*x for x in range(0,5)]

# Number of Primary Devices
users=[2,4,6,8,10]

def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [[float(num) for num in line.strip('[]\n').split(',')] for line in lines]
    return data

def plot_rewards(ax, data, algo_names, plot_type, color, linestyle):
    markers = ['^', 'o', '*', 'd', 's']
    for i, algo_rewards in enumerate(data):
        marker = markers[i % len(markers)]
        ax.plot(distance2, algo_rewards, label=f'{algo_names[i]} - {plot_type}', linestyle=linestyle, marker=marker, color=color)

if __name__ == "__main__":
    # File paths
    linear_file = 'linear_power.txt'
    non_linear_file = 'non_linear_power.txt'
    #linear_N_file='linear_N_final.txt'
    #non_linear_N_file='non_linear_N_final.txt'
    #linear_D_file='linear_D_final.txt'
    #non_linear_D_file='non_linear_D_final.txt'

    # Read data from files
    linear_data = read_data(linear_file)
    non_linear_data = read_data(non_linear_file)
    #linear_N=read_data(linear_N_file)
    #non_linear_N=read_data(non_linear_N_file)
    #linear_D=read_data(linear_D_file)
    #non_linear_D=read_data(non_linear_D_file)
    #print(linear_data)
    #Data=[linear_data,non_linear_data]
    #print(Data)

    '''
    linear_data=[[sum(linear_data[i])/len(linear_data[i]) for i in range(0,len(linear_data),5)],
                 [sum(linear_data[i])/len(linear_data[i]) for i in range(1,len(linear_data),5)],
                 [sum(linear_data[i])/len(linear_data[i]) for i in range(2,len(linear_data),5)],
                 [sum(linear_data[i])/len(linear_data[i]) for i in range(3,len(linear_data),5)],
                 [sum(linear_data[i])/len(linear_data[i]) for i in range(4,len(linear_data),5)]]

    non_linear_data=[[sum(non_linear_data[i])/len(non_linear_data[i]) for i in range(0,len(non_linear_data),5)],
                    [sum(non_linear_data[i])/len(non_linear_data[i]) for i in range(1,len(non_linear_data),5)],
                    [sum(non_linear_data[i])/len(non_linear_data[i]) for i in range(2,len(non_linear_data),5)],
                    [sum(non_linear_data[i])/len(non_linear_data[i]) for i in range(3,len(non_linear_data),5)],
                    [sum(non_linear_data[i])/len(non_linear_data[i]) for i in range(4,len(non_linear_data),5)]]
    '''

    non_linear_data=[[non_linear_data[i][0] for i in range(0,len(non_linear_data),5)],
                 [non_linear_data[i][0] for i in range(1,len(non_linear_data),5)],
                 [non_linear_data[i][0] for i in range(2,len(non_linear_data),5)],
                 [non_linear_data[i][0] for i in range(3,len(non_linear_data),5)],
                 [non_linear_data[i][0] for i in range(4,len(non_linear_data),5)]]
    linear_data=[[linear_data[i][0] for i in range(0,len(linear_data),5)],
                [linear_data[i][0] for i in range(1,len(linear_data),5)],
                [linear_data[i][0] for i in range(2,len(linear_data),5)],
                [linear_data[i][0] for i in range(3,len(linear_data),5)],
                [linear_data[i][0] for i in range(4,len(linear_data),5)]]

# Your list of lists


# Specify the file name
    #file_name_1 = 'data.m'

# Save the list of lists as a MATLAB file

    #scipy.io.savemat(file_name_1, {'data': Data})

    #scipy.io.savemat(file_name_2, {'data': non_linear_data})
    '''
    
    file = open('linear_energy_final.txt','a')
    for item in linear_data:
      file.write(str(item))
      file.write("\n")
    file.close()
    
    file = open('non_linear_energy_final.txt','a')
    for item in non_linear_data:
      file.write(str(item))
      file.write("\n")
    file.close()
    '''
    
    # Algorithm names
    algo_names = ["PER-DDPG", "CER-DDPG", "DDPG", "TD3", "PPO"]

    # Create a figure and left y-axis
    fig, ax1 = plt.subplots()

    # Plot non-linear rewards on the left y-axis
    plot_rewards(ax1, non_linear_data, algo_names, 'Non-Linear', color='tab:blue', linestyle='-')

    # Create a twin y-axis on the right
    ax2 = ax1.twinx()

    # Plot linear rewards on the right y-axis
    plot_rewards(ax2, linear_data, algo_names, 'Linear', color='tab:red', linestyle='--')

    # Set labels and legend for both axes
    ax1.set_xlabel("Distance from BS (RCSD)")
    ax1.set_ylabel("Average Power Transmitted (RCSD) (W) - Non-Linear", color='tab:blue')
    ax2.set_ylabel("Average Power Transmitted (RCSD) (W) - Linear", color='tab:red')

    # Display the legend outside the plot
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.1),ncol=3)
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1.1),ncol=3)

    # Show the plot
    plt.show()
