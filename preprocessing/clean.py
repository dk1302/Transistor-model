import os
import pandas as pd
from sort import sorted_directory_listing_with_os_listdir
import csv

files = sorted_directory_listing_with_os_listdir(os.getcwd()+'/output')

counter = -1
batch = 0
deleted = []

for filename in files:
    if(batch%3 == 0):
        counter += 1
        batch = 0
    if(os.path.getsize(f'output/{filename}') < 22000):
        deleted.append(filename)
        print(filename)
    batch += 1

with open('removed.csv', 'w') as g:
    write = csv.writer(g)
    write.writerow(deleted)