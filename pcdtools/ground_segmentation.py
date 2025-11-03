import numpy as np
import open3d as o3d
import pypatchworkpp
import sys
from pcdtools.io import remove_pothole_points, add_pothole_points_back
from pcdtools.robust_pothole_filter import filter_pothole_candidates

def dump_defaults():
    p = pypatchworkpp.Parameters()
    # pybind11 objects may not have __dict__, so read attributes via dir()
    items = []
    for name in dir(p):
        if name.startswith("_"):
            continue
        try:
            val = getattr(p, name)
        except Exception:
            continue
        items.append((name, val))
    for k, v in sorted(items):
        print(f"{k} = {v}")

dump_defaults()

def set_if_exists(obj, **kwargs):
    for k, v in kwargs.items():
        if hasattr(obj, k):
            setattr(obj, k, v)

# 1) Load the PCD
# pcd = o3d.io.read_point_cloud("/Users/alimuratov/Desktop/indoScan/roads/road-1/segment-000/seg000_cropped.pcd")     
pcd = o3d.io.read_point_cloud("/Users/alimuratov/Downloads/seg000_cropped.pcd")        # Nx3 (x,y,z)

road_pcd, pothole_pts, pothole_cols = remove_pothole_points(pcd, red_threshold=0.7)
pts = np.asarray(road_pcd.points, dtype=np.float32)

params = pypatchworkpp.Parameters()
set_if_exists(params,
uprightness_thr=0.95, 
max_range=1000.0,
num_iter=20)
print([a for a in dir(params) if not a.startswith("_")])  # see actual names in *your* build
pp = pypatchworkpp.patchworkpp(params)
pp.estimateGround(np.ascontiguousarray(pts))  

ground = pp.getGround()
nonground = pp.getNonground()
time_taken = pp.getTimeTaken()

ground_idx = pp.getGroundIndices()
nonground_idx = pp.getNongroundIndices()

print("Origianl Points  #: ", pts.shape[0])
print("Ground Points    #: ", ground.shape[0])
print("Nonground Points #: ", nonground.shape[0])
print("Time Taken : ", time_taken / 1000000, "(sec)")
print("Press ... \n")
print("\t H  : help")
print("\t N  : visualize the surface normals")
print("\tESC : close the Open3D window")

# Get centers and normals for patches
centers = pp.getCenters()
normals = pp.getNormals()

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

# Build ground point cloud using original colors
gi = np.asarray(ground_idx, dtype=int)
orig_pts = np.asarray(road_pcd.points)
ground_pts_src = orig_pts[gi] if gi.size > 0 else np.empty((0, 3), dtype=float)
ground_o3d = o3d.geometry.PointCloud()
ground_o3d.points = o3d.utility.Vector3dVector(ground_pts_src)

orig_colors = np.asarray(road_pcd.colors)
if orig_colors.size > 0 and gi.size > 0:
    ground_cols_src = orig_colors[gi]
    ground_o3d.colors = o3d.utility.Vector3dVector(ground_cols_src.astype(float))

filtered_pothole_pts, filtered_pothole_cols, _ = filter_pothole_candidates(
    pothole_pts, pothole_cols, ground_o3d,
    # coarse (plane-gating)
    k=50, radius=1.0, min_neighbors=15,
    max_above=0.15,            # cap above plane (tail-lights etc.)
    max_below=0.35,            # cap below plane (deep cavities)
    mad_factor=3.5, irls_iters=3,
    # fine (micro cleanup inside pothole sets)
    micro_eps=0.08, micro_min_points=8,
    sor_neighbors=20, sor_std=2.0,
    rad_radius=0.05, rad_min=6
)

# add pothole points back
ground_o3d = add_pothole_points_back(ground_o3d, filtered_pothole_pts, filtered_pothole_cols)
o3d.io.write_point_cloud("ground_cropped_colored_filtered.pcd", ground_o3d)
