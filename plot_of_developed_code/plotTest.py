import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick




rows = []

with open('../overlap_curvature.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row)

print("Total no. of rows: %d"%(csvreader.line_num))


d = rows
df = pd.DataFrame(data = d)
df  = df.astype(float)

cols = range(0,27)  # [0, 1, 2, 3, ..., 25, 26]
df['sum'] = df[cols].sum(axis=1)

#print(df)

writeme = df['sum']

#print(writeme)



k = 0
with open('../overlap_curvature_h2.csv', 'r') as read_obj, \
        open('Overlap_curvature_Sum.csv', 'w', newline='') as write_obj:

    csvreader = csv.reader(read_obj)
    csvwriter = csv.writer(write_obj)


    for row in csvreader:
        print(writeme[k])
        row.append(writeme[k])
        csvwriter.writerow(row)
        k = k + 1


rows = []

with open('Overlap_curvature_Sum.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row)

df = pd.DataFrame(data = rows)

sum_of_alpha = df[28]
curvature_Mycode = df[27]
print('This is the curvature :\n', curvature_Mycode)
print('\nThis is the sum of a :\n', sum_of_alpha)
print(type(sum_of_alpha))

curvature_Mycode = curvature_Mycode.tolist()
curvature_Mycode = list(map(float, curvature_Mycode))
sum_of_alpha = sum_of_alpha.tolist()
sum_of_alpha = list(map(float, sum_of_alpha))

print('The type of sum_of_alpha is ', type(sum_of_alpha[0]))


plt.plot(curvature_Mycode, sum_of_alpha, 'o', color='black')
plt.title('Sum of a over $\kappa_{mycode}$  for h = 2')
plt.xlabel(r'$\kappa_\mathrm{Mycode}$')
plt.ylabel(r'$\Sigma \alpha$')
plt.savefig('Sum_alpha_over_K_mycode_for h_2.pdf')


