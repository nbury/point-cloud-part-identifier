import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import numpy as np
import networkx as nx
import random
from scipy.spatial import ConvexHull as CH
from sklearn.cluster import KMeans
from colorsys import rgb_to_hsv
import struct
import pywavefront as wave
import os
import json
def randomColor():
    return [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))
def length(v):
  return math.sqrt(dotproduct(v, v))
def dist(v1,v2):
    ret = 0
    for i,j in zip(v1,v2):
        ret = ret + (i-j)**2
    return math.sqrt(ret)
def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration >= total: 
        print()
def get_average_point(vec_set):
    ret = []
    for i in range(len(vec_set[0])):
        x = [v[i] for v in vec_set]
        ret.append(sum(x)/len(vec_set))
    return  ret
def unique(list1): 
    # intilize a null list 
    unique_list = [] 
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list
def intersect(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3
#files 
random.seed()
name = "dumbell"
in_file = 'pcd files/dumbell.ply'
orig_file = './'+name+'/orig.xyz'
out_file1 = './'+name+'/out.xyz'
out_file2 = './'+name+'/parts.xyz'
cluster_file ='./'+ name+'/clusters.xyz'
if not os.path.exists('./'+ name):
    os.makedirs(name)
#tweak these to adjust how intercepts/ adjacency is found
show_plots = False
point_per_clust = 20
distance_ratio_threshold = 1.4
sphericity_threshold = .05
radius_buffer = 1

print("Loading PCD")

point_cloud = np.loadtxt(in_file, skiprows=17, max_rows=12000)
num_clusters = int(len(point_cloud) / point_per_clust)
point_data = point_cloud[:,:3]
'''
print("Converting to XYZRGB")
for i in range(len(point_data)):
    #scale data up a bit for ease of use
    point_data[i][0] = point_data[i][0]*100
    point_data[i][1] = point_data[i][1]*100
    point_data[i][2] = point_data[i][2]*100
    #conversion to rgb, check this soon
    rgb = point_data[i][3]
    ba = bytearray(struct.pack("f", rgb))
    r = ba[0]
    g = ba[1]
    b = ba[2]
    h = rgb_to_hsv(r,g,b)[0]
    point_data[i][3] = h


#write pcd file in .xyz format
w = open(orig_file,'w')
w.write("X Y Z R G B\n")
#switch to append mode for points
w = open(orig_file,'a')
colors = []
for p in range(len(point_cloud)):
    point = point_cloud[p]
    rgb = point[3]
    ba = bytearray(struct.pack("f", rgb))
    r = ba[0]
    g = ba[1]
    b = ba[2]
    w.write("{} {} {} {} {} {}\n".format(point[0],point[1],point[2],r,g,b))
    printProgressBar(p,len(point_cloud)-1)
'''

print("Running Kmeans with "+str(num_clusters)+" clusters")
kmeans = KMeans(n_clusters=num_clusters, random_state=random.randint(0,1000)).fit(point_data)
labels = kmeans.labels_

#create list of points for each cluster for calculationsa
cluster_points = []
cluster_labels = []
print("Configuring cluster lists")
for i in range(num_clusters):
    cluster_points.append([])
    cluster_labels.append([])
for p in range(len(point_data)):
    point = point_data[p]
    l = labels[p]
    cluster_points[l].append(point)
    cluster_labels[l].append(p)

#create cluster adjacency graph
print("Finding intercept clusters")
cag = nx.Graph()
cag.add_nodes_from(unique(labels))
adjacent_clusters = []
clust_centers = []
clust_radii = []
sphere_intercept_clusters = []
dist_intercept_clusters = []
clust_data = []
for c in range(len(cluster_points)):
    #find center of each cluster, as well as its endpoints using a convex hull
    clust = cluster_points[c]
    label = labels[cluster_labels[c][0]]
    clust_hull = CH(clust, qhull_options="QJ")
    endpointsIndeces = clust_hull.vertices
    avgPoint = get_average_point(clust)
    clust_centers.append(avgPoint)
    max_dist = 0
    #use endpoints to calculate radius of cluster
    for e in endpointsIndeces:
        point = clust[e]
        if dist(point,avgPoint) > max_dist:
            max_dist = dist(point,avgPoint)
    #Test 1: Distance ratios

    #create graph of points in the cluster
    clust_graph = nx.Graph()
    for i in range(len(clust)):
        clust_graph.add_node(i)
    for i in range(len(clust)):
        for n in range(i,len(clust)):
            clust_graph.add_edge(i,n, dist= dist(clust[i],clust[n]))

    #create sub-graph of close points to use for geodesic distances
    sub_graph = [(u,v,d) for (u,v,d) in clust_graph.edges(data=True) if d['dist'] <=max_dist / 3]
    clust_graph = nx.Graph()
    clust_graph.add_edges_from(sub_graph)
    max_ratio = 0

    #calculate both euclidean and geodesic distances for each pair of endpoints
    for e1 in endpointsIndeces:
        p1 = clust[e1]
        for e2 in endpointsIndeces:
            if e1 != e2 and nx.has_path(clust_graph,source=e1,target=e2):
                p2 = clust[e2]
                path= nx.shortest_path(clust_graph,source=e1,target=e2,weight='dist')
                geo_dist = 0
                for i in range((len(path)-1)):
                    d = dist(clust[path[i]],clust[path[i+1]])
                    geo_dist = geo_dist + d
                euclid_dist = dist(p1,p2)
                ratio = geo_dist/euclid_dist
                if ratio > max_ratio:
                    max_ratio = ratio
    #if the ratio is above the threshold mark it as an intercept cluster
    if(max_ratio > distance_ratio_threshold and label not in dist_intercept_clusters):
        dist_intercept_clusters.append(label)
    
    #Test 2: Sphericity

    #add the buffered radius to list of radii for future use
    clust_radii.append(max_dist*radius_buffer)
    
    #calculate the sphericity using volume of the cluster and the volume of the sphere that is created using the center point and that farthest endpoint 
    box_vol = clust_hull.volume
    sphere_vol = 4/3*math.pi * (max_dist**3)
    sphericity = box_vol / sphere_vol
    # if this ratio is above the threshold, mark it as an intercept cluster
    if sphericity > sphericity_threshold:
        sphere_intercept_clusters.append(label)
    #print(str(max_ratio) +"  "+str(sphericity))
    data = {'label':int(label),'sphericity':float(sphericity),'dist_ratio':float(max_ratio),'center':[float(avgPoint[0]),float(avgPoint[1]),float(avgPoint[2])],'adjacent_clusters':[]}
    clust_data.append(data)
    #calculate spherictity
    printProgressBar(c,len(cluster_points)-1)

print("Calculating cluster adjacency graph")
c = range(len(clust_centers))
xs = []
ys = []
zs = []
#for each cluster, create an edge in the graph if the spheres are overlapping
for c1 in c:
    center_1 = clust_centers[c1]
    xs.append(center_1[0])
    ys.append(center_1[1])
    zs.append(center_1[2])
    dist_1 = clust_radii[c1]
    for c2 in c:
        center_2 = clust_centers[c2]
        dist_2 = clust_radii[c2]
        #use triangle inequality to check overlap and create edge if the clusters are adjacent
        if(dist(center_1,center_2) < (dist_1 + dist_2)):
            edge = (c1,c2)
            adjacent_clusters.append(edge)

cag.add_edges_from(adjacent_clusters)
#show plot of centers
if show_plots:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.array(xs),np.array(ys),np.array(zs),zdir = 'z')
    plt.show()
    print("Drawing Adjacency Graph")

    nx.draw(cag)
    plt.show()

    #plot adjacency graph on the  centers plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for e in cag.edges:
        xs = clust_centers[e[0]][0],clust_centers[e[1]][0]
        ys = clust_centers[e[0]][1],clust_centers[e[1]][1]
        zs = clust_centers[e[0]][2],clust_centers[e[1]][2]
        line = plt3d.art3d.Line3D(xs, ys, zs)
        ax.add_line(line)
    plt.show()

for d in clust_data:
    n = nx.all_neighbors(cag,d['label'])
    ns = []
    for i in n:
        ns.append(int(i))
    d['adjacent_clusters'] = ns
print("Writing part differentiated PCD to new file")
w = open(out_file1,'w')
w.write("X Y Z R G B\n")
#switch to append mode for points
w = open(out_file1,'a')
colors = [(255,0,0), (0,255,0),(0,0,255),(255,255,255)]
for p in range(len(point_cloud)):
    point = point_cloud[p]
    l = labels[p]
    while(len(point)<6):
        point = np.append(point,0)
    if l in sphere_intercept_clusters:
        if l in dist_intercept_clusters:
            l = 0
        else:
            l = 1
    elif l in dist_intercept_clusters:
        l = 2
    else:
        l=3
    point[3] = colors[l][0]
    point[4] = colors[l][1]
    point[5] = colors[l][2]
    w.write("{} {} {} {} {} {}\n".format(point[0],point[1],point[2],point[3],point[4],point[5]))
    printProgressBar(p,len(point_cloud)-1)

print("Writing clustered PCD to new file")
#write header to output file
w = open(cluster_file,'w')
w.write("X Y Z R G B\n")
#switch to append mode for points
w = open(cluster_file,'a')
colors = []
for i in cluster_points:
    colors.append(randomColor())
for p in range(len(point_cloud)):
    point = point_cloud[p]
    l = labels[p]
    while(len(point)<6):
        point = np.append(point,0)
    point[3] = colors[l][0]
    point[4] = colors[l][1]
    point[5] = colors[l][2]
    w.write("{} {} {} {} {} {}\n".format(point[0],point[1],point[2],point[3],point[4],point[5]))
    printProgressBar(p,len(point_cloud)-1)

#find intercept clusters and remove them from the cluster adjacency graph
final_intercept = intersect(sphere_intercept_clusters,dist_intercept_clusters)
for n in final_intercept:
    cag.remove_node(n)
parts = nx.connected_components(cag)

#determine number of components/parts in the PCD
num_parts = nx.number_connected_components(cag)
print("Parts found: "+ str(num_parts))
part_list = []
for part in parts:
    part_list.append(part)

#write part differentiated pcd to output file
w = open(out_file2,'w')
w.write("X Y Z R G B\n")
#switch to append mode for points
w = open(out_file2,'a')
for p in range(len(point_cloud)):
    point = point_cloud[p]
    l = labels[p]
    while(len(point)<6):
        point = np.append(point,0)
    for i,part in enumerate(part_list):
        for n in part:
            if l == n:
                l = i
    point[3] = colors[l][0]
    point[4] = colors[l][1]
    point[5] = colors[l][2]
    w.write("{} {} {} {} {} {}\n".format(point[0],point[1],point[2],point[3],point[4],point[5]))
    printProgressBar(p,len(point_cloud)-1)
with open("./"+name+"/log.json", 'w') as outfile:
    json.dump(clust_data, outfile, indent=2)
