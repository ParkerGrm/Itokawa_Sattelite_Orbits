import trimesh

mesh = trimesh.load('data/itokawa_50k.ply')
mesh.export("data/itokawa_50k_ascii.ply", file_type="ply", encoding="ascii")