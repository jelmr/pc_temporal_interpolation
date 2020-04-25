import subprocess
import os
import os.path as osp
import glob
import pathlib
import time

files = glob.glob("*.fbx")
processes = set()
max_processes = 12

for name in files:
	base, extension = osp.splitext(name)

	processes.add(subprocess.Popen([
		"python",
		"/home/jelmer/torch/scripts/obj_to_ply_sceneflow.py",
                base,
		"-n", "2048"
	]))

	if len(processes) >= max_processes:
		os.wait()
		processes.difference_update([p for p in processes if p.poll() is not None])
