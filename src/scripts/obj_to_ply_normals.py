import argparse
import os
import os.path as osp
import glob
import sys

import pathlib
import os.path as osp
from PIL import Image
from collections import defaultdict
import pandas as pd

import open3d
import numpy as np

BASE_DIR = osp.join(osp.dirname(osp.abspath(__file__)), osp.pardir)
sys.path.append(BASE_DIR)
sys.path.append(osp.join(BASE_DIR, 'src'))

from data.dynamic_point_cloud import FileBackedDynamicPointCloud
from downsample_ply import downsample_uniform, downsample
from utils.open3d_util import np_to_open3d_pc

DEFAULT_ASCII = False
DEFAULT_REMOVE = False
DEFAULT_NUM_POINTS = 100_000




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help=f"Base directory of input.",
                        type=str)
    parser.add_argument("--output_dir",
                        help=f"Output directory [default=<same as input_dir>]",
                        type=str,
                        default="")
    parser.add_argument("--ascii",
                        help=f"Writes the output PLY file as ASCII [default={DEFAULT_ASCII}]",
                        action="store_true",
                        default=DEFAULT_ASCII)
    parser.add_argument("--remove", "-r",
                        help=f"Removes the texures, .obj and .mtl file. Also removes <input_dir> if <output_dir> is "
                        f"specified.",
                        action="store_true",
                        default=DEFAULT_REMOVE)
    parser.add_argument("--n", "-n",
                        type=int,
                        help=f"Number of points to sample from the mesh [default={DEFAULT_NUM_POINTS}]",
                        default=DEFAULT_NUM_POINTS)
    return parser.parse_args()


def parse_mtl(file_name):
    materials_file = open(file_name, "r")
    materials_lines = materials_file.read().splitlines()

    materials = defaultdict(dict)
    current_part = None
    for line in materials_lines:
        d = line.split()
        if len(d) >= 2:
            if d[0] == "newmtl":
                current_part = d[1]
            elif current_part:
                materials[current_part][d[0]] = d[1:]

    return materials


textures = {}  # TODO: Move


def get_texture_point(output_dir, materials, part, x, y):
    global textures

    color_factors = materials[part]["Kd"]

    if part not in textures:
        texture_path = osp.join(output_dir, materials[part]["map_Kd"][0])
        texture_img = Image.open(texture_path)
        texture_pix = texture_img.load()
        width, height = texture_img.size
        textures[part] = {"height": height, "width": width, "pixels": texture_pix}

    texture_data = textures[part]
    x_coord = int((x) * texture_data["width"])
    y_coord = int((1-y) * texture_data["height"])
    pixel = texture_data["pixels"][x_coord, y_coord]

    return [color * float(color_factor) for color, color_factor in zip(pixel, color_factors)]


def parse_obj(input_dir, file_name, materials):
    obj_file = open(file_name, "r")
    obj_lines = obj_file.read().splitlines()

    vertices = []
    texture_vertices = []
    faces = []

    current_part = None
    vertex_coords = {}


    faces_pd = pd.DataFrame(columns=['v1','t1', 'v2', 't2', 'v3', 't3'])

    parts = []
    for line in obj_lines:

        d = line.split()
        if len(d) >= 2:
            if d[0] == "usemtl":
                current_part = len(parts) - 1
                parts.append(d[1])
            elif d[0] == "v":
                vertex = list(map(float, d[1:]))
                vertices.append(vertex)
            elif d[0] == "vt":
                texture_vertex = list(map(float, d[1:]))
                texture_vertices.append(texture_vertex)
            elif d[0] == "f":
                if len(d[1:]) > 3:
                    print("WARNING: Polygon with >3 vertices found and IGNORED.")
                    continue #TODO: Split into triangles
                face = []
                for vertex_data in d[1:]:
                    vertex_data_list = vertex_data.split('/')
                    if len(vertex_data_list) > 2 and vertex_data_list[1] != "":
                        vertex = vertex_data_list[0]
                        vertex_index = int(vertex) - 1
                        face.append(vertex_index)

                        vertex_texture = vertex_data_list[1]
                        vertex_texture_index = int(vertex_texture) -1
                        face.append(vertex_texture_index)
                face.append(current_part)
                faces.append(face)


    vertices_pd = pd.DataFrame(vertices, columns=['x','y','z'])
    faces_pd = pd.DataFrame(faces, columns=['v1','t1','v2','t2','v3','t3', 'mtl_part'])
    textures_pd = pd.DataFrame(texture_vertices, columns=['tx', 'ty'])


    return vertices_pd,faces_pd, textures_pd, parts


def get_face_area(v1_xyz, v2_xyz, v3_xyz):
    return 0.5 * np.linalg.norm(np.cross(v2_xyz - v1_xyz, v3_xyz - v1_xyz), axis=1)

def get_normals(v1_xyz, v2_xyz, v3_xyz):
    u = v2_xyz - v1_xyz
    v = v3_xyz - v1_xyz

    xn = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    yn = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    zn = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    norm = (xn ** 2 + yn ** 2 + zn **2) ** (1/2)

    xn = np.expand_dims(xn, axis=-1)
    yn = np.expand_dims(yn, axis=-1)
    zn = np.expand_dims(zn, axis=-1)

    n =  np.concatenate([xn, yn, zn], axis=-1)

    # Normalize normals to 1
    norm = np.expand_dims(norm, axis=-1)
    n  = n / norm

    return n


def sample(vertices, faces, texture_vertices, parts, n, input_dir, materials, random_indices=None, u=None, v=None):
    # vertices = [x,y,z,r,g,b]
    # faces    = [v1, t1, v2, t2, v3, t3] <-- all indices
    # textures = [x, y]

    xyz = vertices[['x', 'y','z']].values
    v1_xyz = xyz[faces["v1"]]
    v2_xyz = xyz[faces["v2"]]
    v3_xyz = xyz[faces["v3"]]

    tex = texture_vertices.values
    v1_tex = tex[faces["t1"]]
    v2_tex = tex[faces["t2"]]
    v3_tex = tex[faces["t3"]]

    areas = get_face_area(v1_xyz, v2_xyz, v3_xyz)
    area_sum = areas.sum()
    probabilities = areas / area_sum
    if random_indices is None:
        random_indices = np.random.choice(range(len(areas)), size=n, p=probabilities)

    v1_xyz = v1_xyz[random_indices]
    v2_xyz = v2_xyz[random_indices]
    v3_xyz = v3_xyz[random_indices]
    v1_tex = v1_tex[random_indices]
    v2_tex = v2_tex[random_indices]
    v3_tex = v3_tex[random_indices]

    faces = faces.values[random_indices]

    if u is None:
        u = np.random.rand(n, 1)
    if v is None:
        v = np.random.rand(n, 1)

    is_a_problem = u + v > 1
    u[is_a_problem] = 1 - u[is_a_problem]
    v[is_a_problem] = 1 - v[is_a_problem]

    w = 1 - (u + v)

    u_= u.reshape(-1)
    v_ = v.reshape(-1)
    w_ = w.reshape(-1)

    x = u_ * v1_tex[..., 0] + v_ * v2_tex[..., 0] + w_ * v3_tex[..., 0]
    y = u_ * v1_tex[..., 1] + v_ * v2_tex[..., 1] + w_ * v3_tex[..., 1]
    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    xy = np.concatenate([x,y], axis=-1)

    result_xyz = (v1_xyz * u) + (v2_xyz * v) + (w * v3_xyz)
    result_xyz = result_xyz.astype(np.float32)


    result_rgb = get_texture_vertex_colors(xy, faces, parts, input_dir, materials)

    result_normals = get_normals(v1_xyz, v2_xyz, v3_xyz)


    return result_xyz, result_rgb, result_normals, random_indices, u, v


def get_texture_vertex_colors(texture_vertices, faces, parts, input_dir, materials):

    res = []
    for (face, row) in zip(faces, texture_vertices):
        part_idx = face[-1]
        part = parts[part_idx+1]
        x = row[0]
        y = row[1]
        r, g, b = get_texture_point(input_dir,
                                    materials,
                                    part,
                                    x,
                                    y)
        res.append([r,g,b])
    return np.asarray(res) / 255


def color_points(vertices, faces, texture_colors):
    v1_rgb = texture_colors[[faces['t1']]]
    v2_rgb = texture_colors[[faces['t2']]]
    v3_rgb = texture_colors[[faces['t3']]]

    return v1_rgb, v2_rgb, v3_rgb



def convert(input_dir, obj_file_name_1, mtl_file_name_1, obj_file_name_2, mtl_file_name_2, ply_file_name, n, write_ascii=True, u=None, v=None, random_indices=None):
    materials_1 = parse_mtl(mtl_file_name_1)
    vertices_1, faces_1, textures_1, parts_1 = parse_obj(input_dir, obj_file_name_1, materials_1)

    materials_2 = parse_mtl(mtl_file_name_2)
    vertices_2, faces_2, textures_2, parts_2 = parse_obj(input_dir, obj_file_name_2, materials_2)

    if u is None or v is None or random_indices is None:
        points_1, colors_1, normals, random_indices, u, v = sample(vertices_1, faces_1, textures_1, parts_1, n, input_dir, materials_1)
    else:
        points_1, colors_1, normals, random_indices, u, v = sample(vertices_1, faces_1, textures_1, parts_1, n, input_dir, materials_1, u=u, v=v, random_indices=random_indices)


    pc = open3d.PointCloud()
    pc.points = open3d.Vector3dVector(points_1)
    pc.colors = open3d.Vector3dVector(colors_1)
    pc.normals = open3d.Vector3dVector(normals)
    open3d.write_point_cloud(ply_file_name, pc, write_ascii=write_ascii)
    # open3d.draw_geometries([pc])

    return random_indices, u, v


def convert_all(input_dir, output_dir, n, write_ascii=True):
    print(f"=== Converting all .OBJ files in '{input_dir}' to PLY.")
    obj_pattern = osp.join(input_dir, "*.obj")
    files = sorted(glob.glob(obj_pattern))

    u,v,random_indices = None, None, None

    for (i, obj_file_1) in enumerate(files[:-1]):
        obj_file_2 = files[i+1]
        print(f'[{i:05}]: Converting {obj_file_1} (with {obj_file_2}) to PLY.')
        obj_base_name_1, _ = osp.splitext(obj_file_1)
        mtl_file_1 = f'{obj_base_name_1}.mtl'

        obj_base_name_2, _ = osp.splitext(obj_file_2)
        mtl_file_2 = f'{obj_base_name_2}.mtl'

        if osp.exists(mtl_file_1) and osp.exists(mtl_file_2):
            _, base_name = osp.split(obj_base_name_1)
            ply_file = osp.join(output_dir, f'{base_name}.ply')
            if u is None or v is None or random_indices is None:
                random_indices, u, v = convert(input_dir, obj_file_1, mtl_file_1, obj_file_2, mtl_file_2, ply_file, n, write_ascii=write_ascii)
            else:
                random_indices, u, v = convert(input_dir, obj_file_1, mtl_file_1, obj_file_2, mtl_file_2, ply_file, n, write_ascii=write_ascii, u=u, v=v, random_indices=random_indices)
        else:
            print(f"[Error] MTL file {mtl_file_1} and/or {mtl_file_2} does not exist, skipping.")



def remove_pattern(input_dir, pattern):
    path_pattern = osp.join(input_dir, pattern)
    for f in glob.glob(path_pattern):
        os.remove(f)


if __name__ == "__main__":
    # Make open3d shut up when reading/writing point clouds
    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Warning)

    args = parse_args()

    # Make output_dir same as input_dir if not specified
    output_dir = args.input_dir if args.output_dir == "" else args.output_dir

    # Ensure output_dir exists
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convert OBJ -> PLY
    convert_all(args.input_dir, output_dir, n=args.n, write_ascii=args.ascii)

    # Remove all .obj and .mtl files if flag is set
    if args.remove:
        remove_pattern(args.input_dir, "*.obj")
        remove_pattern(args.input_dir, "*.mtl")
        remove_pattern(args.input_dir, "textures/*")
        os.rmdir(osp.join(args.input_dir, "textures"))

    # TODO: Create DPC file
