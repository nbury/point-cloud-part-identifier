from pcd_processor_obj import pcd_part_identifier as pcd_finder
opt = {
    'point_per_clust':50,
    'distance_ratio_threshold':2,
    'sphericity_threshold' :.035,
    'planar_threshold' :.12,
    'radius_buffer':1 
}
pf = pcd_finder(pcd_file = "pcd files/hammer.ply",options = opt, name = "hammer")
#attempt at poisson vector field creation
#pf.plot_vectors()

#graphs from before
#pf.plot_cluster_adjacency()
#pf.plot_centers()
#pf.plot_centers_adjacency()
pf.write_parts()
#run multiple times to see common intecept points (this takes a while)
print("Starting Batch Intercept Points")
pf.batch_intercept_points(25)