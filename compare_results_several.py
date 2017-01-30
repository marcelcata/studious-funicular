# This program plots the results from two different files, comparing them
# IMPORTANT: They have to be trained with the same database percentage to be a fair comparison

# In the command line it has to be used as (example):
# python compare_results_several.py NA_NC_B_N-128__28_12-03-00/results-2017-01-28--03-55-39.txt
# results-2017-01-28--03-55-39.txt

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import sys

fontP = FontProperties()
fontP.set_size('small')

subindex = {}
filename = {}
goals = {}

delimiter = " "
N_ITER = 1000

for i in range(1,len(sys.argv)):
    filename[i] = str(sys.argv[i])
    subindex[i] = raw_input("Subindex measurements {}: ".format(i))
    goals[i] = []
    with open(filename[i], 'r') as f:
        for line in f:
            goals[i].append(line.split(delimiter)[0])
    if N_ITER > len(goals[i]):
        N_ITER = len(goals[i])

for i in range(1,len(sys.argv)):
    plt.plot(range(1, N_ITER+1), goals[i][0:N_ITER], label=subindex[i])

ax = plt.subplot(111)
ax.set_xlim([1, N_ITER])
ax.set_ylim([0,100])
ax.set_ylabel("% Goals")
ax.set_xlabel("Number of epochs")

# Shrink current axis by 20%
box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='lower right')

output_name = raw_input("Name of the plot file (put as many info as possible): ")
strplot = output_name + ".png"
plt.savefig("Comparison_plots" + "/" +strplot)