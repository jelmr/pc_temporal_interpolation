import subprocess
import os
import os.path as osp
import glob
import pathlib
import time

files = glob.glob("*.fbx")
processes = set()
max_processes = 1 # Not really needed to set higher than 1. Blender multithreads itself already.

for name in files:
	base, extension = osp.splitext(name)
	pathlib.Path(base).mkdir(parents=True, exist_ok=True)

	processes.add(subprocess.Popen([
		"blender",
		"--background",
		"-P",
		"/home/jelmer/torch/scripts/fbx_to_obj.py",
		name,
		base
	]))

	if len(processes) >= max_processes:
		os.wait()
		processes.difference_update([p for p in processes if p.poll() is not None])
