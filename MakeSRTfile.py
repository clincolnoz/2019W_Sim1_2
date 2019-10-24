# %%
import csv
import os
import time

# %% os shit
ROOT = os.getcwd()
print(ROOT)
filename = 'SIM12 Ground Truth//Muppets-02-01-01.csv'
# %% processing
with open(os.path.join(ROOT,filename),'r') as file:
    for row in file:
        print(row)