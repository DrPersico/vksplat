import pycolmap, numpy as np

recon = pycolmap.Reconstruction(r"E:\vksplat_data\livingroom_combined\sparse\1")

positions = {}
for img in recon.images.values():
    pos = np.array(img.projection_center())
    positions[img.image_id] = (pos, img.camera_id, img.name)

all_pos = np.array([v[0] for v in positions.values()])
centroid = all_pos.mean(axis=0)
dists = np.linalg.norm(all_pos - centroid, axis=1)

p50 = np.percentile(dists, 50)
p90 = np.percentile(dists, 90)
p95 = np.percentile(dists, 95)
p99 = np.percentile(dists, 99)
print(f"Distance percentiles: 50th={p50:.1f}, 90th={p90:.1f}, 95th={p95:.1f}, 99th={p99:.1f}, max={dists.max():.1f}")

threshold = 15000
within = sum(1 for d in dists if d <= threshold)
print(f"Within {threshold}: {within}/{len(dists)} images")

for cid in [1, 2, 3]:
    cam_dists = []
    for i, (pos, c, name) in enumerate(positions.values()):
        if c == cid:
            cam_dists.append(dists[i])
    cam_dists = sorted(cam_dists)
    n = len(cam_dists)
    within_c = sum(1 for d in cam_dists if d <= threshold)
    worst5 = [f"{d:.0f}" for d in cam_dists[-5:]]
    print(f"Camera {cid}: {within_c}/{n} within {threshold}, worst 5: {worst5}")
