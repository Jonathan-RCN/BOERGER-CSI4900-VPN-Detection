import numpy as np

data=[np.array([0.95986204, 0.34092347]), np.array([0.96914904, 0.8520949 ]), np.array([0.90925529, 0.70772059]), np.array([0.91033991, 0.77480916]), np.array([0.929587  , 0.94892168]), np.array([0.98936662, 0.99665392]), np.array([0.97936726, 0.97772021]), np.array([0.9933898 , 0.98587699]), np.array([0.95840722, 0.51457627]), np.array([0.89948707, 0.15551617])]
sum0=0
sum1=0
l0=[]
l1=[]

for ele in data:
    print (ele)
    print(ele[0])
    print(ele[1])
    sum0+=ele[0]
    sum1+=ele[1]
    l0.append(ele[0])
    l1.append(ele[1])

print(sum0/len(data))
print(np.average(l0))
print(sum1/len(data))
print(np.average(l1))

