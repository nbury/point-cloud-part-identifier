'''
object to process and identify parts in a point cloudt
functions:
    init
    calculate
    write parts
    write orig (xyzrgb format)
    write clusters
    write identified clusters
    each funct should have verbose mode
    write data to pcd_log.json
    plot centers/cag/cag on centers
'''
import nvidia_smi
nvidia_smi.nvmlInit()
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print(f"Total GPU Memory: {info.total:,d}")
print(f"Free GPU Memory: {info.free:,d}")
print(f"Used GPU Memory: {info.used:,d}")
for g in gpus:
    print(g)
print("\n\n\n\n\n\n\n")
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import numpy as np
import networkx as nx
import random
from scipy.spatial import ConvexHull as CH
from scipy.spatial import KDTree as KD
import kmeanstf
from kmeanstf import KMeansTF as kmeans
import struct
import threading
import os
import json
from skspatial.objects import Plane
from multiprocessing.pool import ThreadPool as Pool

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

class pcd_part_identifier:
    def __init__(self,pcd_file,pcd_options = {},name="pcd_out",options ={}):
        #set up io files
        self.name = name
        self.in_file = pcd_file
        self.orig_file = name+'/orig.xyz'
        self.out_file1 = name+'/out.xyz'
        self.part_file = name+'/parts.xyz'
        self.cluster_file = name+'/clusters.xyz'
        self.data_file = name+'/log.json'
        self.intercept_points_file = name+'/intercept_points.xyz'
        self.format = ""
        if not os.path.exists('./'+ self.name):
            os.makedirs(self.name)
        
        #load pcd data
        self.max_rows = 500000 if 'max_rows' not in pcd_options else pcd_options['max_rows']
        self.skip = 50 if 'skiprows' not in pcd_options else pcd_options['skiprows']
        self.load_pcd(self.in_file)
        self.point_data = self.point_cloud[:,:3]
        self.kd_tree =KD(self.point_data)
        self.point_vectors = []
        #self.calc_point_vectors()
        self.point_intercept_count = [0]*len(self.point_data)
        #settings
        self.point_per_clust = 50 if 'point_per_clust' not in options else options['point_per_clust']
        self.distance_ratio_threshold = 2.5 if 'distance_ratio_threshold' not in options else options['distance_ratio_threshold']
        self.sphericity_threshold = .01 if 'sphericity_threshold' not in options else options['sphericity_threshold']
        self.planar_threshold = .1 if 'planar_threshold' not in options else options['planar_threshold']
        self.radius_buffer = 1.1 if 'radius_buffer' not in options else options['radius_buffer']
        
        
        print("Finding Parts")
        self.find_parts()
        
        print("Writing Outputs")
        self.write_parts(self.part_file)
        self.write_intercept_clusters(self.cluster_file)
        self.write_data(self.data_file)
    def load_pcd(self, file):

        self.point_cloud = np.loadtxt(self.in_file, skiprows=self.skip, max_rows=self.max_rows)
        for x in range(len(self.point_cloud)):
            for y in range(3):
                self.point_cloud[x][y] = self.point_cloud[x][y]

    def check_intercept_cluster(self,c):
        #find center of each cluster, as well as its endpoints using a convex hull
        clust = self.cluster_points[c]
        label = self.labels[self.cluster_labels[c][0]]
        clust_hull = CH(clust, qhull_options="QJ")
        endpointsIndeces = clust_hull.vertices
        avgPoint = get_average_point(clust)
        self.clust_centers.append(avgPoint)
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
        if(max_ratio > self.distance_ratio_threshold and label not in self.dist_intercept_clusters):
            self.dist_intercept_clusters.append(label)
        
        #Test 2: Sphericity

        #add the buffered radius to list of radii for future use
        self.clust_radii.append(max_dist*self.radius_buffer)
        
        #calculate the sphericity using volume of the cluster and the volume of the sphere that is created using the center point and that farthest endpoint 
        box_vol = clust_hull.volume
        sphere_vol = 4/3*math.pi * (max_dist**3)
        sphericity = box_vol / sphere_vol
        # if this ratio is above the threshold, mark it as an intercept cluster
        if sphericity > self.sphericity_threshold:
            self.sphere_intercept_clusters.append(label)
        #print(str(max_ratio) +"  "+str(sphericity))
        
        #calculate spherictity
        #printProgressBar(c,len(self.cluster_points)-1)

        #Test 3: planar distance
        clust_plane = Plane.best_fit(clust)
        avg_planar_dist = 0
        for p in clust:
            avg_planar_dist = avg_planar_dist + clust_plane.distance_point(p)
        avg_planar_dist = (avg_planar_dist / len(clust) )/max_dist
        if avg_planar_dist > self.planar_threshold:
            self.planar_intercept_clusters.append(label)
        data = {'label':int(label),'sphericity':float(sphericity),'planar_dist':avg_planar_dist,'dist_ratio':float(max_ratio),'center':[float(avgPoint[0]),float(avgPoint[1]),float(avgPoint[2])],'adjacent_clusters':[]}
        self.clust_data.append(data)
    def find_parts(self):
        self.num_clusters = int(len(self.point_cloud) / self.point_per_clust)
        print("Num Clusters: "+str(self.num_clusters))
        #print("Running Kmeans with "+str(self.num_clusters)+" clusters")
        self.kmeans = kmeans(n_clusters=self.num_clusters, random_state=random.randint(0,1000)).fit(self.point_data)
        print("Kmeans Complete")
        self.labels = self.kmeans.labels_
        self.cag = nx.Graph()
        self.cag.add_nodes_from(unique(self.labels))
        self.adjacent_clusters = []
        self.clust_centers = []
        self.clust_radii = []
        self.sphere_intercept_clusters = []
        self.dist_intercept_clusters = []
        self.clust_data = []
        self.planar_intercept_clusters = []

        #create list of points for each cluster for calculationsa
        self.cluster_points = []
        self.cluster_labels = []
        for i in range(self.num_clusters):
            self.cluster_points.append([])
            self.cluster_labels.append([])
        for p in range(len(self.point_data)):
            point = self.point_data[p]
            l = self.labels[p]
            self.cluster_points[l].append(point)
            self.cluster_labels[l].append(p)
        pool_size = 10
        pool = Pool(pool_size)

        for c in range(len(self.cluster_points)):
            pool.apply_async(self.check_intercept_cluster, (c,))

        pool.close()
        pool.join()
        #print("Calculating cluster adjacency graph")

        c = range(len(self.clust_centers))
        #for each cluster, create an edge in the graph if the spheres are overlapping
        for c1 in c:
            center_1 = self.clust_centers[c1]
            dist_1 = self.clust_radii[c1]
            for c2 in c:
                center_2 = self.clust_centers[c2]
                dist_2 = self.clust_radii[c2]
                #use triangle inequality to check overlap and create edge if the clusters are adjacent
                if(dist(center_1,center_2) < (dist_1 + dist_2)):
                    edge = (c1,c2)
                    self.adjacent_clusters.append(edge)

        self.cag.add_edges_from(self.adjacent_clusters)
        self.cag_parts = self.cag.copy()
        self.intercept_clusters = self.planar_intercept_clusters
        #self.intercept_clusters = intersect(self.planar_intercept_clusters,self.dist_intercept_clusters)
        for n in self.intercept_clusters:
            self.cag_parts.remove_node(n)
        self.num_parts = nx.number_connected_components(self.cag_parts)
        self.parts = nx.connected_components(self.cag_parts)
        print("Parts found: "+ str(self.num_parts))
        self.part_list = []
        for part in self.parts:
            self.part_list.append(part)

    def calc_point_vectors(self):
        for i,point in enumerate(self.point_data):
            #find center of each cluster, as well as its endpoints using a convex hull
            dists, indeces = self.kd_tree.query(point,10)
            points = []
            for i in range(1,len(indeces)-1):
                points.append(self.point_data[indeces[i]])
            plane = Plane.best_fit(points)
            self.point_vectors.append(plane.project_vector(point))
            printProgressBar(i,len(self.point_cloud)-1)

    def batch_intercept_points(self,num_runs):
        for run in range(num_runs):
            printProgressBar(run,num_runs-1)
            self.find_parts()
            for p in range(len(self.point_cloud)):
                label = self.labels[p]
                if label in self.intercept_clusters:
                    self.point_intercept_count[p] = self.point_intercept_count[p]+1
        #print(self.point_intercept_count)
        #print("Writing Intercept Points to new file")
        #write header to output file
        w = open(self.intercept_points_file,'w')
        w.write("X Y Z R G B\n")
        #switch to append mode for points
        w = open(self.intercept_points_file,'a')
        for p in range(len(self.point_cloud)):
            point = self.point_data[p]
            count = self.point_intercept_count[p]
            c =float(count / num_runs)
            w.write("{} {} {} {} {} {}\n".format(point[0],point[1],point[2],int(c*255),int(c*255),int(c*255)))
    def plot_vectors(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = []
        y = []
        z = []
        u = []
        v = []
        w = []
        for i,vec in enumerate(self.point_vectors):
            l = length(vec)
            if(random.randint(0,100)>70):
                x.append(self.point_data[i][0])
                y.append(self.point_data[i][1])
                z.append(self.point_data[i][2])
                u.append(vec[0])
                v.append(vec[1])
                w.append(vec[2])
        ax.quiver(x, y, z, u, v, w,length= .05,arrow_length_ratio = .3)
        plt.autoscale()
        plt.show()
    def write_clusters(self,file='_out.xyz'):
        #print("Writing clustered PCD to new file")
        #write header to output file
        w = open(self.cluster_file,'w')
        w.write("X Y Z R G B\n")
        #switch to append mode for points
        w = open(self.cluster_file,'a')
        colors = []
        for i in self.cluster_points:
            colors.append(randomColor())
        for p in range(len(self.point_cloud)):
            point = self.point_cloud[p]
            l = self.labels[p]
            while(len(point)<6):
                point = np.append(point,0)
            point[3] = colors[l][0]
            point[4] = colors[l][1]
            point[5] = colors[l][2]
            w.write("{} {} {} {} {} {}\n".format(point[0],point[1],point[2],point[3],point[4],point[5]))
            #printProgressBar(p,len(self.point_cloud)-1)

    def write_parts(self,file='_out.xyz'):
        w = open(self.part_file,'w')
        w.write("X Y Z R G B\n")
        #switch to append mode for points
        w = open(self.part_file,'a')
        colors = []
        for i in self.cluster_points:
            colors.append(randomColor())
        for p in range(len(self.point_cloud)):
            point = self.point_cloud[p]
            l = self.labels[p]
            while(len(point)<6):
                point = np.append(point,0)
            stop = True
            for i,part in enumerate(self.part_list):
                
                for n in part:
                    if l == n and stop:
                        l = i
                        stop = False
            if stop:
                point[3] = 255
                point[4] = 255
                point[5] = 255
            else:
                point[3] = colors[l][0]
                point[4] = colors[l][1]
                point[5] = colors[l][2]
            w.write("{} {} {} {} {} {}\n".format(point[0],point[1],point[2],point[3],point[4],point[5]))
            #printProgressBar(p,len(self.point_cloud)-1)
    def write_intercept_clusters(self,file='_out.xyz'):
        #print("Writing part differentiated PCD to new file")
        w = open(self.out_file1,'w')
        w.write("X Y Z R G B\n")
        #switch to append mode for points
        w = open(self.out_file1,'a')
        colors = [(255,0,0), (0,255,0),(0,0,255),(255,255,255)]
        for p in range(len(self.point_cloud)):
            r = 0
            g = 0
            b = 0
            point = self.point_cloud[p]
            l = self.labels[p]
            while(len(point)<6):
                point = np.append(point,0)

            if l in self.planar_intercept_clusters:
                r = 255
            if l in self.sphere_intercept_clusters:
                g = 0
            if l in self.dist_intercept_clusters:
                b = 255
            
            point[3] = r
            point[4] = g
            point[5] = b
            w.write("{} {} {} {} {} {}\n".format(point[0],point[1],point[2],point[3],point[4],point[5]))
            #printProgressBar(p,len(self.point_cloud)-1)
    def write_data(self,file="_log.json"):
        with open("./"+self.name+"/log.json", 'w') as outfile:
            json.dump(self.clust_data, outfile, indent=2)
    def plot_centers(self):
        xs = []
        ys = []
        zs = []
        c = range(len(self.clust_centers))
        for c1 in c:
            center_1 = self.clust_centers[c1]
            xs.append(center_1[0])
            ys.append(center_1[1])
            zs.append(center_1[2])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.array(xs),np.array(ys),np.array(zs),zdir = 'z')
        plt.show()
    def plot_cluster_adjacency(self):
        nx.draw(self.cag)
        plt.show()
    def plot_centers_adjacency(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for e in self.cag.edges:
            xs = self.clust_centers[e[0]][0],self.clust_centers[e[1]][0]
            ys = self.clust_centers[e[0]][1],self.clust_centers[e[1]][1]
            zs = self.clust_centers[e[0]][2],self.clust_centers[e[1]][2]
            line = plt3d.art3d.Line3D(xs, ys, zs)
            ax.add_line(line)
        plt.show()


