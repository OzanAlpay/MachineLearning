import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



#Load the data and separate it into three arrays; one for each class
data = np.loadtxt('iris.data')
data0 = data[data[:,4 ]==0]
data1 = data[data[:,4 ]==1]
data2 = data[data[:,4 ]==2]
pi = 22/7
# ^^ YOUR CODE HERE ^^

# Plot each type of data for all classes in 1D (with shifts of 0.1 for better visualization)
fig = plt.figure()
#plt.plot(data0[:,0], np.ones(len(data0[:,0]))*0.0, '+r', label='Data 0 Class 0')
#plt.plot(data1[:,0], np.ones(len(data1[:,0]))*0.1, '+g', label='Data 0 Class 1')
#plt.plot(data2[:,0], np.ones(len(data2[:,0]))*0.2, '+b', label='Data 0 Class 2')
#
#plt.plot(data0[:,1], np.ones(len(data0[:,1]))*1.0, 'xr', label='Data 1 Class 0')
#plt.plot(data1[:,1], np.ones(len(data1[:,1]))*1.1, 'xg', label='Data 1 Class 1')
#plt.plot(data2[:,1], np.ones(len(data2[:,1]))*1.2, 'xb', label='Data 1 Class 2')
#
#plt.plot(data0[:,2], np.ones(len(data0[:,2]))*2.0, '.r', label='Data 2 Class 0')
#plt.plot(data1[:,2], np.ones(len(data1[:,2]))*2.1, '.g', label='Data 2 Class 1')
#plt.plot(data2[:,2], np.ones(len(data2[:,2]))*2.2, '.b', label='Data 2 Class 2')
#
#plt.plot(data0[:,3], np.ones(len(data0[:,3]))*3.0, '1r', label='Data 3 Class 0')
#plt.plot(data1[:,3], np.ones(len(data1[:,3]))*3.1, '1g', label='Data 3 Class 1')
#plt.plot(data2[:,3], np.ones(len(data2[:,3]))*3.2, '1b', label='Data 3 Class 2')
#
#plt.legend(fontsize=9, loc=3)
#plt.show()
#Examining the plots above select two of the data types and plot them in 2D - one data type for each axis. Let's say you chose ath and bth columns as your data. This means you have to plot dataN[:,a] vs dataN[:,b] for N=0,1,2.



plt.plot(data0[:,2], data0[:,3], '.r', label='Data 0 Class 0')
plt.plot(data1[:,2], data1[:,3], '.g', label='Data 0 Class 1')
plt.plot(data2[:,2], data2[:,3], '.b', label='Data 0 Class 2')
plt.legend(fontsize=9, loc=3)
plt.show()



#  Using the two datatype you have chosen, extract the 2D Gaussian (Normal) distribution parameters. Numpy functions are called here to be used  for validation of your results.
mx0 = np.mean(data0[:,2])
print mx0
my0 = np.mean(data0[:,3])
print my0
cov0 = np.cov(data0[:,2:4].T)
print cov0
mx1 = np.mean(data1[:,2])
my1 = np.mean(data1[:,3])
cov1 = np.cov(data1[:,2:4].T)
mx2 = np.mean(data2[:,2])
my2 = np.mean(data2[:,3])
cov2 = np.cov(data2[:,2:4].T)

print "Previously Calculated Means are : "
print mx0,mx1,mx2,my0,my1,my2
print "Previously Calculated Variances and Covariances are : "
print cov0
print cov1
print cov2
print "Means which are calculated by me are : "
def calculateMean(data,num):
    return np.sum(data[:,num])/len(data[:,2])
for num in range(2,4):
        print calculateMean(data0,num)
        print calculateMean(data1,num)
        print calculateMean(data2,num)
totalMeanX = calculateMean(data,2)
totalMeanY = calculateMean(data,3)
print "totalMeanX : ",totalMeanX
print "totalMeanY : ",totalMeanY
print "Standart Deviations which are calculated by me are : "
def calculateStandartDeviation(data,num):
    myVar = 0
    mean = calculateMean(data,num)
    for numInner in range(0,len(data[:,num])):
        myVar += (data[numInner:numInner+1,num]-mean)**2
    return myVar/(len(data[:,num])-1)
    
for num in range(2,4):
    print calculateStandartDeviation(data0,num)
    print calculateStandartDeviation(data1,num)
    print calculateStandartDeviation(data2,num)
    
def calculateCovariance(data):
    myCov = 0
    meanX = calculateMean(data,2)
    meanY = calculateMean(data,3)
    for numInner in range(0,len(data[:,2])):
        myCov += (data[numInner:numInner+1,2]-meanX)*(data[numInner:numInner+1,3]-meanY)
    return myCov/(len(data[:,2])-1)

print "Covariances which are calculated by me are : "
print calculateCovariance(data0)
print calculateCovariance(data1)
print calculateCovariance(data2)
totalStandartDeviationX = calculateStandartDeviation(data,2)
print "TotalStandartDeviationX : ", totalStandartDeviationX
totalStandartDeviationY = calculateStandartDeviation(data,3)
print "TotalStandartDeviationY : ", totalStandartDeviationY
print "TotalCovariance : "
totalCovariance = calculateCovariance(data)
print totalCovariance
totalCovarianceMatrix = np.matrix([[totalStandartDeviationX[0],totalCovariance[0]],[totalCovariance[0],totalStandartDeviationY[0]]])
print "Total Covariance Matrix : "
print totalCovarianceMatrix

print "VALUES HOLDS"
# Part 5: Plot the Gaussian surfaces for each class.
## First, we generate the grid to compute the Gaussian function on.
vals = np.linspace(np.min(data),np.max(data) , 500)
x,y = np.meshgrid(vals, vals)


def gaussian_2d(x,y,mx,my,cov): 
    ''' x and y are the 2D coordinates to calculate the function value
        mx and my are the mean parameters in x and y axes
        cov is the 2x2 variance-covariance matrix'''
    varianceX=cov[0][0] #Variance of x
    varianceY=cov[1][1] #Variance of y
    covariance=cov[0][1] #co-varaince of this set which equal to [0][1]
    p=covariance/(np.sqrt(varianceX)*np.sqrt(varianceY))
    ret=(1/(2*pi*np.sqrt(varianceX)*np.sqrt(varianceY)))*(1/np.sqrt(1-np.power(p,2)))*np.exp((-1/(2*(1-np.power(p,2))))*((np.power((x-mx),2)/varianceX)+(np.power((y-my),2)/varianceY)-(2*p*(x-mx)*(y-my))/(np.sqrt(varianceX)*np.sqrt(varianceY))))
    #i copied this equation from github
    return ret
##compute the Gaussian function outputs for each entry in our mesh an plot the surface for each class.
z0 = gaussian_2d(x, y, mx0, my0, cov0)
z1 = gaussian_2d(x, y, mx1, my1, cov1)
z2 = gaussian_2d(x, y, mx2, my2, cov2)
fig0 = plt.figure()
ax0 = fig0.add_subplot(111, projection='3d')
ax0.plot_surface(x, y, z0, cmap=cm.jet, linewidth=0, antialiased=False)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(x, y, z1, cmap=cm.jet, linewidth=0, antialiased=False)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(x, y, z2, cmap=cm.jet, linewidth=0, antialiased=False)



# Part 6: Classify each sample in the dataset based on your findings and assign a class label. Explain your reasoning behind your implementation with few sentences

lbl = []
def findMaximum(zd0,zd1,zd2):
    
    if((zd0 > zd1) and  (zd0 > zd2)):
        return 0
    elif(zd1 > zd2):
        return 1
    else:
        return 2
print totalCovarianceMatrix
def addLabels(data,lbl):
    for d in data:
        #label = 0
        zd0 = gaussian_2d(d[2], d[3], mx0, my0, cov0)
        zd1 = gaussian_2d(d[2], d[3], mx1, my1, cov1)
        zd2 = gaussian_2d(d[2], d[3], mx2, my2, cov2)
        lbl.append(findMaximum(zd0,zd1,zd2))
        #Now we will put all of them into our gaussian_2d again
        #get data [now our gaussian trained] and look for the maximum value
        #maximum value == maximum chance to assign right class
        
        
addLabels(data,lbl)   
success_rate = 0
#Now we should compare our lbl[elementindex] and data[:,4
def FindPerformance(data,lbl):
    j=0
    score=0
    for d in data:
        if(d[4]==lbl[j]):
            score += 1
            print "SCORE ",d[4],lbl[j],score,j
        j+=1      
    print "Total Score : ",score
    success_rate = (score*100)/len(data)
    print "Success Rate is : ",success_rate
FindPerformance(data,lbl)            
#Repeat the same process for non-overlapping training and test sets.
print "PART 8 STARTS HERE"
data_test = np.vstack((data[0:25],data[50:75],data[100:125]))
data_train = np.vstack((data[25:50],data[75:100],data[125:150]))
data_test0 = data_test[data_test[:,4]==0]
data_test1 = data_test[data_test[:,4]==1]
data_test2 = data_test[data_test[:,4]==2]
data_train0 = data_train[data_train[:,4]==0]
data_train1 = data_train[data_train[:,4]==1]
data_train2 = data_train[data_train[:,4]==2]
# First I need to find Means,Variance,Covariance etc for our new train dataset
dataTrainMx0 = calculateMean(data_train0,2)
dataTrainMy0 = calculateMean(data_train0,3)
dataTrainMx1 = calculateMean(data_train1,2)
dataTrainMy1 = calculateMean(data_train1,3)
dataTrainMx2 = calculateMean(data_train2,2)
dataTrainMy2 = calculateMean(data_train2,3)
totalMeanTrainX = calculateMean(data_train,2)
totalMeanTrainY = calculateMean(data_train,3)
dataTrainCov0 = np.cov(data_train0[:,2:4].T)
dataTrainCov1 = np.cov(data_train1[:,2:4].T)
dataTrainCov2 = np.cov(data_train2[:,2:4].T)
dataTrainCov = np.cov(data_train[:,2:4].T)
vals = np.linspace(np.min(data_train),np.max(data_train) , 500)
x,y = np.meshgrid(vals, vals)
z0 = gaussian_2d(x, y, dataTrainMx0, dataTrainMy0, dataTrainCov0)
z1 = gaussian_2d(x, y, dataTrainMx2, dataTrainMy1, dataTrainCov1)
z0 = gaussian_2d(x, y, dataTrainMx2, dataTrainMy2, dataTrainCov2)
#now our learning function trained.
lbl=[]
addLabels(data_test,lbl)
FindPerformance(data_test,lbl)




