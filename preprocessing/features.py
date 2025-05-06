import csv
import os
from sort import sorted_directory_listing_with_os_listdir

files = sorted_directory_listing_with_os_listdir(os.getcwd()+'/output')

batch = 0

for filename in files:
    if(batch%3 == 0):
        batch = 0

    lines = []
    with open(os.path.join(os.getcwd()+'/output', filename), 'r') as file:
        for line in file:
            processed_line = line.strip()
            lines.append(processed_line)

        lines = lines[25:121]

        size = (97)
        output = [0 for i in range(size)]
        counter = 0
        lineCounter = 1
        scale = False

        for line in lines:
            counter = 0
            for word in line.split():
                counter += 1
                if( counter%13 == 0 ):
                    scaleValue = ''
                    for char in word:
                        if(char == '-'):
                            scale=True
                            continue
                        if(scale):
                            scaleValue += char
                    rawValue = word[:-5]
                    if(lineCounter == 1):
                        output[0] = scaleValue
                    else:
                        output[lineCounter] = word
                    lineCounter += 1

    formatted = [0 for k in range(size)]
    scale = [0 for i in range(2)]
    counter = 0

    for x in output:
        if(counter == 0):
            temp= int(x)
            formatted[counter] = float(temp)
            scale[0] = formatted[counter]
            scale[1] = formatted[counter]
        else:
            formatted[counter] = float(x)
        counter += 1

    if(batch == 0):
        with open('features_v1.csv', 'a', newline='') as g:
            write = csv.writer(g)
            write.writerow(formatted[1:])
        with open('scale_v1.csv', 'a', newline='') as g:
            write = csv.writer(g)
            write.writerow(scale)
    elif(batch == 1):
        with open('features_v5.csv', 'a', newline='') as g:
            write = csv.writer(g)
            write.writerow(formatted[1:])
        with open('scale_v5.csv', 'a', newline='') as g:
            write = csv.writer(g)
            write.writerow(scale)
    elif(batch == 2):
        with open('features_v10.csv', 'a', newline='') as g:
            write = csv.writer(g)
            write.writerow(formatted[1:])
        with open('scale_v10.csv', 'a', newline='') as g:
            write = csv.writer(g)
            write.writerow(scale)
    
    batch += 1

