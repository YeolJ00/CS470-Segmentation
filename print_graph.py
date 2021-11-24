import matplotlib.pyplot as plt
import pandas as pd

logfile = 'logs/deeplabv3plus_vit_resnet101_voc_os16.out'
df = pd.read_csv(logfile)

plt.figure(figsize=(18,4))
plt.subplot(1, 3, 1)
data = df[['Itrs','Loss']]
x = data.Itrs
y = data.Loss
plt.plot(x,y)
plt.title('Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(1, 3, 2)
data = df[['Itrs','Overall_Acc']]
data = data.dropna()
x = data.Itrs
y = data.Overall_Acc
plt.plot(x,y)
plt.title('[Val] Overall Acc')
plt.xlabel('Iteration')
plt.ylabel('Acc')

plt.subplot(1, 3, 3)
data = df[['Itrs','Mean_IoU']]
data = data.dropna()
x = data.Itrs
y = data.Mean_IoU
plt.plot(x,y)
plt.title('[Val] Mean IoU')
plt.xlabel('Iteration')
plt.ylabel('IoU')

plt.show()
