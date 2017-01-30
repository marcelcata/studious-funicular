# This program plots the results from two different files, comparing them
# IMPORTANT: They have to be trained with the same database percentage to be a fair comparison

# In the command line it has to be used as (example):
# python compare_results.py NA_NC_B_N-128__28_12-03-00/results-2017-01-28--03-55-39.txt results-2017-01-28--03-55-39.txt

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import sys

fontP = FontProperties()
fontP.set_size('small')

# filename1 = raw_input("Directory and name of first txt file: ")
# filename2 = raw_input("Directory and name of second txt file: ")
filename1 = str(sys.argv[1])
filename2 = str(sys.argv[2])
subindex1 = raw_input("Subindex first measurements: ")
subindex2 = raw_input("Subindex second measurements: ")
output_name = raw_input("Name of the plot file (put as many info as possible): ")

delimiter = " "

# Open first file
Goals1 = []
Subs1 = []
Ins1 = []
Borr1 = []

with open(filename1,'r') as f:
    for line in f:
        Goals1.append(line.split(delimiter)[0])
        Subs1.append(line.split(delimiter)[1])
        Ins1.append(line.split(delimiter)[2])
        Borr1.append(line.split(delimiter)[3])

# Open second file
Goals2 = []
Subs2 = []
Ins2 = []
Borr2 = []

with open(filename2,'r') as f:
    for line in f:
        Goals2.append(line.split(delimiter)[0])
        Subs2.append(line.split(delimiter)[1])
        Ins2.append(line.split(delimiter)[2])
        Borr2.append(line.split(delimiter)[3])

N_ITER = min(len(Goals1), len(Goals2))

plt.plot(range(1, N_ITER+1), Goals1[0:N_ITER], label="Goals_"+subindex1)  # Goals
plt.plot(range(1, N_ITER+1), Subs1[0:N_ITER], label="Subs_"+subindex1)  # Substitutions
plt.plot(range(1, N_ITER+1), Ins1[0:N_ITER], label="Ins_"+subindex1)  # Insertions
plt.plot(range(1, N_ITER+1), Borr1[0:N_ITER], label="Borr_"+subindex1)  # Borrades

plt.plot(range(1, N_ITER+1), Goals2[0:N_ITER], label="Goals_"+subindex2)  # Goals
plt.plot(range(1, N_ITER+1), Subs2[0:N_ITER], label="Subs_"+subindex2)  # Substitutions
plt.plot(range(1, N_ITER+1), Ins2[0:N_ITER], label="Ins_"+subindex2)  # Insertions
plt.plot(range(1, N_ITER+1), Borr2[0:N_ITER], label="Borr_"+subindex2)  # Borrades

ax = plt.subplot(111)
ax.set_xlim([1, N_ITER])
ax.set_ylim([0,100])
ax.set_ylabel("%")
ax.set_xlabel("Number of epochs")

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = fontP)

strplot = output_name + ".png"
plt.savefig("Comparison_plots" + "/" +strplot)