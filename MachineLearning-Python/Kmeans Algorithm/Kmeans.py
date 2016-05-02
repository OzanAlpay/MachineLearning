import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Part 1: Load the data
data = np.loadtxt('iris.data')

# Part 2: Plot 1st, 3rd and 4th features in 3D
fig_size = (30,20)
marker_size = 60
fig = plt.figure(figsize=fig_size)
ax = fig.add_subplot(111, projection='3d')
s = ax.scatter(data[:,0],data[:,2],data[:,3], marker='o', c='r', s=marker_size)
s.set_edgecolors = s.set_facecolors = lambda *args:None

#This function calculates distance between two points in 3-d
#with (x1-x2)^2 + (y1-y2)^2 ^ (z1-z2)^2 method
def getDistance(centroid,sample):
    #print "Get Distance Sample : "
    #print sample;
    #print "Selected Centroid : "
    #print centroid;
    totalDistance = (sample[0]-centroid[0])**2 + (sample[2]-centroid[1])**2 + (sample[3]-centroid[2])**2;
    #print "Total Distance :"
    #print totalDistance
    return totalDistance;
#Calculates total cost of kmeans with current centroids,and clusters    
def calc_cost(centroids, cluster_ids, data):
    dist = []
    for i in range(len(data)):
        sample = data[i];
        cluster_id = cluster_ids[i];
        centroid = centroids[cluster_id];
        dist.append(getDistance(centroid,sample));
    return sum(dist)


# 
# Random initialization for cluster centroids within data range
def init_centroids(k, data):
    init_center = []
    Xranges = [min(data[:,0]),max(data[:,0])];
    Yranges = [min(data[:,2]),max(data[:,2])];
    Zranges = [min(data[:,3]),max(data[:,3])];
    #print "X range minimum : %f, maximum : %f "%(Xranges[0],Xranges[1]);
    #print "Y range minimum : %f, maximum : %f "%(Yranges[0],Yranges[1]);
    #print "Z range minimum : %f, maximum : %f "%(Zranges[0],Zranges[1]);        
    #randomArray= [np.random.uniform(Xranges[0],Xranges[1]),np.random.uniform(Yranges[0],Yranges[1]),np.random.uniform(Zranges[0],Zranges[1])];
    while k>0:
        init_center.append([np.random.uniform(Xranges[0],Xranges[1]),np.random.uniform(Yranges[0],Yranges[1]),np.random.uniform(Zranges[0],Zranges[1])]);
        k=k-1
    #print "RandomArray : ";
    #print init_center;
    return np.array(init_center)
centroids = init_centroids(3,data);
#print "First randomly selected centroids are : "
#print centroids;
#Cluster assignments for data samples
def assign_cluster(centroids, data):
    cluster_id = []
    for sample in data:
        tempArray = [];
        for i in range(len(centroids)):
            tempArray.append(((sample[0]-centroids[i,0])**2) + ((sample[2]-centroids[i,1])**2) + ((sample[3]-centroids[i,2])**2));
            #tempArray.append(((sample[:,0]-centroids[i,0])**2) + ((sample[:,2]-centroids[i,2])**2) + ((sample[:,3]-centroids[i,3])**2));
            
        cluster_id.append(tempArray.index(min(tempArray)));
    #print "Cluster id length : %d" % len(cluster_id);
    return np.array(cluster_id)
cluster_ids = assign_cluster(centroids,data);
print "Calculation Cost : "
print calc_cost(centroids,cluster_ids,data);
#print cluster_ids;
#Recalculate cluster centroids
def calc_centroids(cluster_ids, data):
    centroids = [];
    num_of_elements = [];
    cluster_id_set = set(cluster_ids);
    #print cluster_id_set;
    
    for element in cluster_id_set:
        #print element;
        num_of_elements.append(0);
        centroids.append([0,0,0]);
    for cluster_id in cluster_ids:
        #print "Cluster id : %d" %cluster_id
        #print "Before number of elements in this cluster : %d" %num_of_elements[cluster_id];
        num_of_elements[cluster_id] = num_of_elements[cluster_id] + 1;
        
        #print "After number of elements in this cluster : %d"%num_of_elements[cluster_id];
    #print "Num of elements : ";
    #print num_of_elements;
    for i in range(len(data)):
        cluster_id = cluster_ids[i];
        sample = data[i];
        #print "Sample : ";
        #print sample;
        #print "Sample[0] : ";
        #print sample[0];
        centroids[cluster_id][0] = centroids[cluster_id][0] + sample[0];
        centroids[cluster_id][1] = centroids[cluster_id][1] + sample[2];
        centroids[cluster_id][2] = centroids[cluster_id][2] + sample[3];
    
    #print "Total of centroids :";
    #print centroids;
    for cluster_id in cluster_id_set:
        centroids[cluster_id][0] = centroids[cluster_id][0] / num_of_elements[cluster_id];
        centroids[cluster_id][1] = centroids[cluster_id][1] / num_of_elements[cluster_id];
        centroids[cluster_id][2] = centroids[cluster_id][2] / num_of_elements[cluster_id];
    #print "New Centroids : ";
    #print centroids;
    return np.array(centroids)
centroids = calc_centroids(cluster_ids,data);


#Calculation of cost (optimization objective)
print "Calculation Cost : "
print calc_cost(centroids,cluster_ids,data);
# TODO
#Part 4: Implement the function that runs k-means on the data with 50 random initializations. For all trials, calculate and record cost values at each iteration and final cluster assignments.
def run_kmeans(k, data):
    cost_array_all = []
    cost_final_all = []
    cluster_ids_all =[]
    # 
    ind = np.argmin(cost_final_all)
    return cluster_ids_all[ind], cost_array_all[ind]
# TODO    
#  Run the k-means algorithm with k between 1 and 6. Plot the final cost vs k curve and observe the elbow to determine the best k.
plt.figure(figsize=fig_size)
plt.xlabel('k', fontsize=32)
plt.ylabel('Optimized Cost', fontsize=32)
#


# TODO
# With the best k value, run k-means on the data. Plot the cost vs iteration curve and show 1st, 3rd and 4th features in 3D using different colors for different clusters.
plt.figure(figsize=fig_size)
plt.xlabel('Iteration No.', fontsize=32)
plt.ylabel('Cost', fontsize=32)
# YOUR CODE HERE