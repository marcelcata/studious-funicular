# This program plots the results from two different files, comparing them
# IMPORTANT: They have to be trained with the same database percentage to be a fair comparison

# In the command line it has to be used as (example):
# python compare_results.py results-2017-01-28--03-55-39.txt results-2017-01-28--03-55-39.txt file_name_plot

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import sys

delimiter = " "

# Open first file
filename = str(sys.argv[1])

Goals1 = []
Subs1 = []
Ins1 = []
Borr1 = []

with open(filename,'r') as f:
    for line in f:
        Goals1.append(line.split(delimiter)[0])
        Subs1.append(line.split(delimiter)[1])
        Ins1.append(line.split(delimiter)[2])
        Borr1.append(line.split(delimiter)[3])

# Open second file
filename = str(sys.argv[2])

Goals2 = []
Subs2 = []
Ins2 = []
Borr2 = []

with open(filename,'r') as f:
    for line in f:
        Goals2.append(line.split(delimiter)[0])
        Subs2.append(line.split(delimiter)[1])
        Ins2.append(line.split(delimiter)[2])
        Borr2.append(line.split(delimiter)[3])

N_ITER = min(len(Goals1), len(Goals2))

plt.plot(range(1, N_ITER+1), Goals1[0:N_ITER], label="Goals")  # Goals
plt.plot(range(1, N_ITER+1), Subs1[0:N_ITER], label="Subs")  # Substitutions
plt.plot(range(1, N_ITER+1), Ins1[0:N_ITER], label="Ins")  # Insertions
plt.plot(range(1, N_ITER+1), Borr1[0:N_ITER], label="Borr")  # Borrades

plt.plot(range(1, N_ITER+1), Goals2[0:N_ITER], label="Goals_bid")  # Goals
plt.plot(range(1, N_ITER+1), Subs2[0:N_ITER], label="Subs_bid")  # Substitutions
plt.plot(range(1, N_ITER+1), Ins2[0:N_ITER], label="Ins_bid")  # Insertions
plt.plot(range(1, N_ITER+1), Borr2[0:N_ITER], label="Borr_bid")  # Borrades

plt.axis([1, N_ITER, 0, 100])
plt.ylabel("%")
plt.legend(loc='upper left')
plt.xlabel("Number of epochs")

strplot = "plot-" + str(sys.argv[3]) + ".png"
plt.savefig(strplot)