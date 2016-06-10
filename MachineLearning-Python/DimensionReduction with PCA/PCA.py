# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 11:51:40 2016

@author: ozan
@mail : ozanalpay at yandex dot com dot tr
"""

#CENG 463 : Introduction to Machine Learning Homework #3
#190201012 -- Ozan Alpay



#Part 1 Read Data : 
import numpy as np

data = np.loadtxt('glass.data',delimiter=',');

#Part 2 Calculate Covariance Matrix :
#Now we its time to construct covariance matrix
#To do this i created some helper functions 
#calculateMean , calculateVariance and calculateCovariance
#Then i created an empty array size of 10
#And in last step of this part i filled in with data

#A function to calculate Mean
def calculateMean(data):
    return sum(data)/len(data);
"""
def calculateVariance(data,mean):
    variance = 0.0;
    for i in range(0,len(data)):
        variance += (data[i]-mean)**2;
    return variance/len(data);
"""    
def calculateCovariance(firstData,secondData,firstMean,secondMean):
    variance = 0.0;
    for i in range(0,len(firstData)):
        variance += (firstData[i]-firstMean) * (secondData[i]-secondMean);
    return variance/len(firstData);
        
covarianceMatrix = np.zeros((9,9));
#I dont get first and last columns of glass.info
#Since they are not relevant 
#print covarianceMatrix;
for i in range(1,10):
    for j in range(1,10):
        #print "Calculated Covariance for i : %s and j : %s " %(i,j);
        tempCovariance =  calculateCovariance(data[:,i],data[:,j],calculateMean(data[:,i]),calculateMean(data[:,j]));
        covarianceMatrix[i-1][j-1] = tempCovariance;
print covarianceMatrix;

#Part 3 Calculating eigenvectors and eigenvalues :

U ,s , V = np.linalg.svd(covarianceMatrix);
print U;
print s;
print V;

#Part 4 Plotting eigenvalues and PoV
#Eigenvalues stored in s
#An empty matrix to store PoV values
from matplotlib import pyplot as plt
povMatrix = [0 for _ in range(9)];
print povMatrix;
#a helper function to calculate PovValues with given k,[like k=2 then (y0+y1+y2)/total]
def findPovValue(k,eigenvalues):
    total = 0.0;
    for i in range(0,k):
        total += eigenvalues[i];
    return total/sum(eigenvalues);
#It starts from 1 to 9 since there isny any logic modelling data with 0 feature
for i in range(1,9):
    povMatrix[i] = findPovValue(i,s);
    print povMatrix[i];    

#figure of Eigenvalues
plt.figure(figsize=(10,10)); 
plt.xlabel('Num', fontsize=32); 
plt.ylabel('Eigenvalues', fontsize=32);
plt.plot(s);

#figure of Proportion of variance
print povMatrix;
plt.figure(figsize=(10,10));
plt.xlabel('Num', fontsize=32);
plt.ylabel('Pov Vals', fontsize=32);
plt.plot(povMatrix); 


#Part 5 Dimesion reduction

#Matrix multiplication function that i am going to use later:
redData = np.matrix(data[0:214,1:10]);
#Now we must reduce Eigenvectors matrix to 9x3 and the transpose it
#So basically 
reducedEigenvectors = U[:3];
#Then find transpose of reducedEigenvectors
reducedEigenvectorsTranspose = np.matrix(np.transpose(reducedEigenvectors));
#reducedEigenvectorstranspose = np.matrix(reducedEigenvectorsTranspose);
result = redData.dot(reducedEigenvectorsTranspose);
print result;



#Part 6 Plot data again [reduced form]
from mpl_toolkits.mplot3d import Axes3D

fig_size = (50,50)
marker_size = 10
fig = plt.figure(figsize=fig_size)
ax = fig.add_subplot(111, projection='3d')
s = ax.scatter(result[:,0],result[:,1],data[:,2],marker='x', c='b', s=marker_size);

s.set_edgecolors = s.set_facecolors = lambda *args:None
 









        

