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
#Then returns it
def getDistance(centroid,sample):
    totalDistance = (sample[0]-centroid[0])**2 + (sample[2]-centroid[1])**2 + (sample[3]-centroid[2])**2;
    return totalDistance;
#Calculates total cost of distance between current centroids and clusters    
def calc_cost(centroids, cluster_ids, data):
    dist = []
    for i in range(len(data)):
        sample = data[i];
        cluster_id = cluster_ids[i];
        centroid = centroids[cluster_id];
        dist.append(getDistance(centroid,sample));
    return sum(dist)
#Solves a problem that occurs in some extraordinary cases for example:
#If in first run of algorithm with k we have only two clusters which are(0,1)
#There isn't any memberof 2 where it should be
#After assign_clusters called there will some data's which changes their cluster from
#0 or 1 to 2
#In this or similar cases my function get ArrayOutOfBounds Exception,
#To handle this I wrote that function
def restore_centroids(num_of_elements,cluster_id_set,cluster_id):
    if cluster_id >= len(cluster_id_set):
        num_of_elements.append(1);
        return num_of_elements;
    else:
        return num_of_elements[:cluster_id].append(1) + num_of_elements[cluster_id];

# Called at start
# Random initialization for cluster centroids within data range
def init_centroids(k, data):
    init_center = []
    Xranges = [min(data[:,0]),max(data[:,0])];
    Yranges = [min(data[:,2]),max(data[:,2])];
    Zranges = [min(data[:,3]),max(data[:,3])];
    while k>0:
        init_center.append([np.random.uniform(Xranges[0],Xranges[1]),np.random.uniform(Yranges[0],Yranges[1]),np.random.uniform(Zranges[0],Zranges[1])]);
        k=k-1
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
#Recalculate cluster centroids to decrease to value of cost function
def calc_centroids(cluster_ids, data):
    centroids = [];
    num_of_elements = [];
    cluster_id_set = set(cluster_ids);
    for element in cluster_id_set: 
        num_of_elements.append(0);
        #I create an array which is for example k = 3 == [0,0,0] at start
        #After that if I find a data with cluster 1 I will increase middle element 1
        #[Because its index 1] and it will be [0,1,0] and goes on like that
    for cluster_id in cluster_ids:
        #Now for every cluster_id i will increase my array appropriate element
        #I created restore_centroids function to handle some extraordinary cases
        #I am going to use that in here
        try :
            #If everything goes well
            num_of_elements[cluster_id] = num_of_elements[cluster_id] + 1;
        except IndexError:
            #If there is any problem
            num_of_elements = restore_centroids(num_of_elements,cluster_id_set,cluster_id);
    for cluster_num in num_of_elements:
        centroids.append([0,0,0]);
    #To find centroid for k=t's x value i used that formula
    #Add all data's x values which belongs to t cluster
    #Then divide it with number of datas which belong to cluster t
    #In the for loops below first i calculated , and then i divide
    for i in range(len(data)):
        cluster_id = cluster_ids[i];
        sample = data[i];
        centroids[cluster_id][0] = centroids[cluster_id][0] + sample[0];
        centroids[cluster_id][1] = centroids[cluster_id][1] + sample[2];
        centroids[cluster_id][2] = centroids[cluster_id][2] + sample[3];
    for cluster_id in cluster_id_set:
        centroids[cluster_id][0] = centroids[cluster_id][0] / num_of_elements[cluster_id];
        centroids[cluster_id][1] = centroids[cluster_id][1] / num_of_elements[cluster_id];
        centroids[cluster_id][2] = centroids[cluster_id][2] / num_of_elements[cluster_id];
    return np.array(centroids)
centroids = calc_centroids(cluster_ids,data);
#Calculation of cost (optimization objective)

print "Calculation Cost : "
print calc_cost(centroids,cluster_ids,data);
# TODO
#Part 4: Implement the function that runs k-means on the data with 50 random initializations. For all trials, calculate and record cost values at each iteration and final cluster assignments.
def run_kmeans(k, data):
    print "K means runs with number of %d cluster " %k;
    cost_array_all = []
    cost_final_all = []
    cluster_ids_all =[]
    for i in range(0,50):
        centroids_for_this_run = init_centroids(k,data);
        cluster_ids_all.append(assign_cluster(centroids_for_this_run,data));
        centroids_for_this_run = (calc_centroids(cluster_ids_all[i],data));
        cost_final_all.append(calc_cost(centroids_for_this_run,cluster_ids_all[i],data));
        cost_array_all.append(centroids_for_this_run);
    ind = np.argmin(cost_final_all)
    print cost_final_all;
    #print "CLUSTER_IDS_ALL";
    #print cluster_ids_all;
    print "MIN COST IS : "
    print min(cost_final_all);
    print "MIN COST IS ANOTHER WAY : "
    print cost_final_all[ind];
    return cluster_ids_all[ind], cost_final_all[ind]
run_kmeans(3,data);
# TODO    
#  Run the k-means algorithm with k between 1 and 6. Plot the final cost vs k curve and observe the elbow to determine the best k.
costs=[];
for k in range (2,6):
    costs.append(run_kmeans(k,data));
print costs;

plt.figure(figsize=fig_size)
plt.xlabel('k', fontsize=32)
plt.ylabel('Optimized Cost', fontsize=32)
# TODO
# With the best k value, run k-means on the data. Plot the cost vs iteration curve and show 1st, 3rd and 4th features in 3D using different colors for different clusters.
plt.figure(figsize=fig_size)
plt.xlabel('Iteration No.', fontsize=32)
plt.ylabel('Cost', fontsize=32)
# YOUR CODE HERE