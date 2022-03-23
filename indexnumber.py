#Task
#write a program to find index number, based on the following conditions:
# 1.Unzip indexNumber.zip from dataset folder.
# 2.Pick one key -cpe23Uri from sourceDF.csv
# 3Find the indexNumber of the above key from “CPEMatchString” columns of TargetDf.csv


import pandas as pd
#Read the Dataset
scr = pd.read_csv('sourceDF.csv')
dest = pd.read_csv('TargetDF.csv',low_memory=False)
u_id = scr['cpe23Uri'][1]



#check the all keys present or not if preent return index number
l = len(scr)
id_dest = []
for i in range(l):
  u_id = scr['cpe23Uri'][i]
  for ind in dest.index:
    if dest['CPEMatchString'][ind] == u_id:
      id_dest.append(ind)

#check the particular key present or not if preent return index number
# u_id = scr['cpe23Uri'][0]
# for ind in dest.index:
#   if dest['CPEMatchString'][ind] == u_id:
#     id_dest.append(ind)
print(id_dest)

#no keys are matching in Targetsource..in'Cpematchstring..To return index number