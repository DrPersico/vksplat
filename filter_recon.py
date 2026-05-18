"""Filter outlier cameras from a COLMAP reconstruction and save a clean copy."""
import pycolmap
import numpy as np
import os
import shutil

src = r"E:\vksplat_data\livingroom_combined\sparse\1"
dst = r"E:\vksplat_data\livingroom_combined\sparse\1_clean"

recon = pycolmap.Reconstruction(src)

positions = {}
for img_id, img in recon.images.items():
    pos = np.array(img.projection_center())
    positions[img_id] = pos

all_pos = np.array(list(positions.values()))
centroid = all_pos.mean(axis=0)
dists = {iid: np.linalg.norm(p - centroid) for iid, p in positions.items()}

median_dist = np.median(list(dists.values()))
threshold = median_dist * 3
print(f"Median distance: {median_dist:.1f}, threshold: {threshold:.1f}")

outlier_ids = [iid for iid, d in dists.items() if d > threshold]
print(f"Outliers: {len(outlier_ids)} / {len(recon.images)}")
for iid in outlier_ids:
    name = recon.images[iid].name
    d = dists[iid]
    print(f"  Removing {name} (dist={d:.0f})")
    frame_id = recon.images[iid].frame_id
    recon.deregister_frame(frame_id)

print(f"Remaining: {recon.num_reg_images()} images, {recon.num_points3D()} points")

os.makedirs(dst, exist_ok=True)
recon.write(dst)
print(f"Saved to {dst}")
