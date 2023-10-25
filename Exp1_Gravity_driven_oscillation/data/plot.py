import matplotlib.pyplot as plt
import os
import csv



def read_csv(filename, location=None, poprows=None, delimiter=None):
    if location == 0:
        filepath = filename
    elif location != None and location != 0:
        filepath = str(location) + str(filename)
    else:
        filepath = str(os.getcwd()) + str('\\') + str(filename)

    if delimiter == '\t':
        delim = '\t'
    else:
        delim = ','

    with open(filepath, 'r') as infile:
        data = csv.reader(infile, delimiter=delim)
        datalist, rows = [], []
        for row in data:
            datalist.append(row)
        if poprows != None:
            for i in range(poprows):
                datalist.pop(0)
        for j in range(len(datalist[0])):
            globals()['string%s' % j] = []
            for k in datalist:
                globals()['string%s' % j].append(float(k[j]))
            rows.append(globals()['string%s' % j])
        infile.close()

    return rows


data7 = read_csv("Set7_1.txt", location=0, poprows=9, delimiter='\t')
data1 = read_csv("Set1.txt", location=0, poprows=3, delimiter='\t')
data2 = read_csv("Set2.txt", location=0, poprows=3, delimiter='\t')
data3 = read_csv("Set3.txt", location=0, poprows=4, delimiter='\t')
data4 = read_csv("Set4.txt", location=0, poprows=6, delimiter='\t')
data5 = read_csv("Set5.txt", location=0, poprows=6, delimiter='\t')
data6 = read_csv("Set6.txt", location=0, poprows=4, delimiter='\t')


#plt.plot(data7[0], data7[1], label='h= 7.6 cm')
#plt.plot(data1[0], data1[1], label='h= 7.1 cm')
#plt.plot(data2[0], data2[1], label='h= 6.5 cm')
#plt.plot(data3[0], data3[1], label='h= 5.8 cm')
#plt.plot(data4[0], data4[1], label='h= 5.0 cm')
#plt.plot(data5[0], data5[1], label='h= 4.4 cm')
plt.plot(data6[0], data6[1], label='h= 3.9 cm')

plt.xlabel("Time (second)")
plt.ylabel("Fluid Height (cm)")
plt.legend()
plt.grid()
plt.show()










