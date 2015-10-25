#k means
#k is the number of clusters

import pandas as pd
import numpy as np
from matplotlib.pyplot import plot,show
from scipy.cluster.vq import kmeans,vq
unData = pd.read_csv('./un.csv')

data1 = pd.DataFrame(unData[['GDPperCapita','infantMortality']])
data1 = data1.dropna()
data2 = pd.DataFrame(unData[['GDPperCapita','lifeMale']])
data2 = data2.dropna()
data3 = pd.DataFrame(unData[['GDPperCapita','lifeFemale']])
data3 = data3.dropna()

#there must be a way to loop through these
km1 = kmeans(data1.values,3) #array of cluster centers
id1 = vq(data1.values,km1[0]) #[0] because it spits out another value
x1 = data1[id1[0]==0]
km2 = kmeans(data1.values,3) #array of cluster centers
id2 = vq(data1.values,km2[0]) #[0] because it spits out another value
x2 = data1[id2[0]==0]
km3 = kmeans(data1.values,3) #array of cluster centers
id3 = vq(data1.values,km3[0]) #[0] because it spits out another value
x3 = data1[id3[0]==0]

plot(x1['GDPperCapita'],x1['infantMortality'],'ob',
	x2['GDPperCapita'],x2['infantMortality'],'or',
	x3['GDPperCapita'],x3['infantMortality'],'og')
show()


km1 = kmeans(data2.values,3) #array of cluster centers
id1 = vq(data2.values,km1[0]) #[0] because it spits out another value
x1 = data2[id1[0]==0]
km2 = kmeans(data2.values,3) #array of cluster centers
id2 = vq(data2.values,km2[0]) #[0] because it spits out another value
x2 = data2[id2[0]==0]
km3 = kmeans(data2.values,3) #array of cluster centers
id3 = vq(data2.values,km3[0]) #[0] because it spits out another value
x3 = data2[id3[0]==0]
plot(x1['GDPperCapita'],x1['lifeMale'],'ob',
	x2['GDPperCapita'],x2['lifeMale'],'or',
	x3['GDPperCapita'],x3['lifeMale'],'og')
show()


km1 = kmeans(data3.values,3) #array of cluster centers
id1 = vq(data3.values,km1[0]) #[0] because it spits out another value
x1 = data3[id1[0]==0]
km2 = kmeans(data3.values,3) #array of cluster centers
id2 = vq(data3.values,km2[0]) #[0] because it spits out another value
x2 = data3[id2[0]==0]
km3 = kmeans(data3.values,3) #array of cluster centers
id3 = vq(data3.values,km3[0]) #[0] because it spits out another value
x3 = data3[id3[0]==0]
plot(x1['GDPperCapita'],x1['lifeFemale'],'ob',
	x2['GDPperCapita'],x2['lifeFemale'],'or',
	x3['GDPperCapita'],x3['lifeFemale'],'og')
show()


