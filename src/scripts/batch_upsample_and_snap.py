import subprocess
import os
import os.path as osp
import glob
import pathlib
import time

files = glob.glob("*.fbx")
command = "python"
processes = set()
max_processes = 12

for name in files:
	base, extension = osp.splitext(name)

	processes.add(subprocess.Popen([
		command,
		"upsample_and_snap.py",
                osp.join("/path/to/low_resolution_interpolated", base),
                osp.join("/path/to/low_resolution_interpolated", base, "frames.dpc"),
                osp.join("/path/to/high_resolution_low_framerate", base, "sceneflow"),
                osp.join("/path/to/high_resolution_low_framerate", base, "sceneflow", "frames.dpc"),
                osp.join("/path/to/output_high_resolution_high_framerate", base)
	]))

	if len(processes) >= max_processes:
		os.wait()
		processes.difference_update([p for p in processes if p.poll() is not None])

