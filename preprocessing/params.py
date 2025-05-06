import csv
import os
from sort import sorted_directory_listing_with_os_listdir

files = sorted_directory_listing_with_os_listdir(os.getcwd()+'/input')

for filename in files:
    lines = []
    with open(os.path.join(os.getcwd()+'/input', filename), 'r') as file:
        for line in file:
            processed_line = line.strip()
            lines.append(processed_line)

        lines = lines[2:19]

        output = [0 for i in range(17)]
        yOutputCounter = 0
        counter = 0

        for line in lines:
            counter = 0
            var = ''
            start = False
            word = line.split()
            for char in word[1]:
                if(char == '='):
                    start = True
                    continue
                if(start):
                    var += char
            output[yOutputCounter] = var
            yOutputCounter += 1

    formatted = [0 for k in range(17)]
    counter = 0

    for x in output:
        formatted[counter] = float(x)
        counter += 1

    with open('new_params.csv', 'a', newline='') as g:
        write = csv.writer(g)
        write.writerow(formatted)
