from pcd_processor_obj import pcd_part_identifier as pcd_finder
opt = {
    'point_per_clust':25,
    'distance_ratio_threshold':1.4,
    'sphericity_threshold' :.05,
    'planar_threshold' :.125,
    'radius_buffer':1 
}
pf = pcd_finder(pcd_file = "pcd files/dumbell.ply",options = opt, name = "dumbell")
#attempt at poisson vector field creation
pf.plot_vectors()

#graphs from before
pf.plot_centers()
pf.plot_centers_adjacency()

#run multiple times to see common intecept points (this takes a while)
pf.batch_intercept_points(50)