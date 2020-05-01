import subprocess
import os
import os.path as osp
import glob
import pathlib
import time

files = glob.glob("*.fbx")
command = "blender"
processes = set()
max_processes = 1 # Do not set to higher than 1 for this script, blender already parallelizes.

for name in files:
	base, extension = osp.splitext(name)
	pathlib.Path(base).mkdir(parents=True, exist_ok=True)

	processes.add(subprocess.Popen([
		command,
		"--background",
		"-P",
		"fbx_to_obj.py",
		name,
		base
	]))

	if len(processes) >= max_processes:
		os.wait()
		processes.difference_update([p for p in processes if p.poll() is not None])
