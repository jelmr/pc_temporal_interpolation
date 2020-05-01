import os
import glob
import re
import os.path as osp
import pathlib
import argparse
import sys

DEFAULT_PREFIX = "frame"


def parse_args():
    parser = argparse.ArgumentParser(prog="blender --background -P fbx_to_obj.py -- ")
    parser.add_argument("script_name",
                        help="Name of the script [fbx_to_obj.py]",
                        type=str)
    parser.add_argument("input",
                        help="FBX file that needs to be converted.",
                        type=str)
    parser.add_argument("output_dir",
                        help="Directory where all .obj files will be written to.",
                        type=str)
    parser.add_argument("--prefix",
                        help="Prefix of output .obj files. [default=?].",
                        type=str,
                        default=DEFAULT_PREFIX)
    return parser.parse_known_args()


def fix_mtl(texture_dir, mtl_file_name):
    # Read a list of all extracted textures
    textures = [fn for fn in os.listdir(texture_dir)]

    mtl_file = open(mtl_file_name, 'r')

    result = ""

    # Replace all known texture occurances by the correct path
    for line in mtl_file:
        for texture in textures:

            pattern = "([^ ]*"+re.escape(texture)+")"
            texture_loc = osp.join(texture_dir, texture)
            match = re.search(pattern, line)
            if match:
                line = re.sub(pattern, texture_loc, line)
                break
            # pattern = f"(.*{re.escape(texture)})"
            # texture_loc = osp.join(texture_dir, texture)
            # texture_str = f"map_Kd {texture_loc}"
            # match = re.search(pattern, line)
            # if match:
            #     line = texture_str
            #     break

        result += line

    mtl_file.close()

    # Overwrite the original MTL by corrected version
    mtl_file = open(mtl_file_name, "w")
    mtl_file.write(result)


def fix_all_mtl(out_dir_name, texture_dir_name):


    for fn in glob.glob("*.mtl"):
        _, mtl_file_name = osp.split(fn)
        fix_mtl(texture_dir_name, mtl_file_name)


if __name__ == "__main__":
    args, _ = parse_args()

    if 'bpy' not in sys.modules:
        print("This script must be run through Blender. Use 'python fbx_to_obj.py -h' for usage.")
        sys.exit(1)

    import bpy

    fbx_fn = args.input

    # Delete the default cube
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Import FBX
    print("=== Importing FBX === ")
    bpy.ops.import_scene.fbx(filepath=fbx_fn)

    # Create output dir if it doesn't exist
    print("=== Creating output dir === ")
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Export .obj files
    print("=== Exporting OBJ files === ")
    output_path = osp.join(args.output_dir, args.prefix+'.obj')
    bpy.ops.export_scene.obj(filepath=output_path, use_animation=True, use_edges=True, use_uvs=True,
                             use_materials=True, use_vertex_groups=True, use_triangles=True)

    # Export textures
    print("=== Exporting textures === ")
    os.chdir(args.output_dir)
    for img in bpy.data.images:
        img.unpack()

    # Fix all MTL files to use proper texture paths
    print("=== Fixing MTL files === ")
    fix_all_mtl(args.output_dir, "textures")
