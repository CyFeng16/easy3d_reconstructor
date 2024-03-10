import copy
import zlib
from pathlib import Path
from typing import List, Union
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.device import to_numpy
from dust3r.utils.image import load_images, rgb
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes

plt.ion()


# Define constants for clarity and easy maintenance
MIN_PORT: int = 10000
MAX_PORT: int = 60000
BATCH_SIZE: int = 1
INIT_METHOD: str = "mst"
LEARNING_RATE: float = 0.01


def string_to_port_crc32(input_string: str) -> int:
    """
    Converts a string to a CRC32 hash and maps it to a port number within a specific range.

    :param input_string: The input string to hash.
    :return: A port number derived from the CRC32 hash of the input string.
    """

    # Calculate the CRC32 hash, ensuring the result is a non-negative integer
    crc32_hash: int = zlib.crc32(input_string.encode()) & 0xFFFFFFFF
    # Map the hash to a port number within the specified range
    port_number: int = MIN_PORT + crc32_hash % (MAX_PORT - MIN_PORT)

    return port_number


def _convert_scene_output_to_glb(
    out_dir: Union[str, Path],
    imgs: np.ndarray,
    pts3d: np.ndarray,
    mask: List[np.ndarray],
    focals: np.ndarray,
    cams2world: np.ndarray,
    cam_size: float = 0.05,
    cam_color: Optional[List[str]] = None,
    as_pointcloud: bool = False,
    transparent_cams: bool = False,
) -> Path:

    # Validate inputs lengths and types
    assert (
        len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    ), "Input arrays have inconsistent lengths."

    # Convert inputs to numpy arrays if they aren't already
    pts3d, imgs, focals, cams2world = map(to_numpy, [pts3d, imgs, focals, cams2world])

    scene = trimesh.Scene()

    if as_pointcloud:
        # Combine and reshape points and colors for full pointcloud
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        # Construct and add geometry for each image/mask pair
        meshes = [
            pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]) for i in range(len(imgs))
        ]
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # Add cameras to the scene
    for i, pose_c2w in enumerate(cams2world):
        color = (
            cam_color[i]
            if isinstance(cam_color, list)
            else cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        )
        add_scene_cam(
            scene,
            pose_c2w,
            color,
            None if transparent_cams else imgs[i],
            focals[i],
            imsize=imgs[i].shape[1::-1],
            screen_width=cam_size,
        )

    # Apply transformation to align with OpenGL coordinate system
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))

    # Export the scene to a .glb file
    out_dir_path = Path(out_dir)
    out_file = out_dir_path / "scene.glb"
    scene.export(file_obj=str(out_file))

    return out_file


def get_3d_model_from_scene(
    outdir: Union[str, Path],
    scene: "Scene",  # Assuming 'Scene' is a custom class, use the actual class name
    min_conf_thr: float = 3.0,
    as_pointcloud: bool = False,
    mask_sky: bool = False,
    clean_depth: bool = False,
    transparent_cams: bool = False,
    cam_size: float = 0.05,
) -> Optional[Path]:
    """
    Extracts a 3D model (glb file) from a reconstructed scene.

    :param outdir: Output directory for the glb file.
    :param scene: The reconstructed scene from which to generate the 3D model.
    :param min_conf_thr: The minimum confidence threshold for point cloud filtering.
    :param as_pointcloud: Flag to output as a point cloud instead of a mesh.
    :param mask_sky: Flag to mask out the sky from the scene.
    :param clean_depth: Flag to apply depth cleaning.
    :param transparent_cams: Flag to render cameras as transparent.
    :param cam_size: Size of the camera representation in the scene.
    :return: The path to the generated glb file.
    """

    if scene is None:
        return None

    # Apply post-processing steps if required
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # Extract necessary data from the scene
    rgb_img = scene.imgs
    focals = to_numpy(scene.get_focals().cpu())
    cams2world = to_numpy(scene.get_im_poses().cpu())
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())

    # Generate and return the 3D model file path
    return _convert_scene_output_to_glb(
        outdir,
        rgb_img,
        pts3d,
        msk,
        focals,
        cams2world,
        cam_size=cam_size,
        as_pointcloud=as_pointcloud,
        transparent_cams=transparent_cams,
    )


def get_reconstructed_scene(
    outdir,
    model,
    device,
    image_size,
    filelist,
    schedule,
    niter,
    min_conf_thr,
    as_pointcloud,
    mask_sky,
    clean_depth,
    transparent_cams,
    cam_size,
    scenegraph_type,
    winsize,
    refid,
):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(
        imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True
    )
    output = inference(pairs, model, device, batch_size=BATCH_SIZE)

    mode = (
        GlobalAlignerMode.PointCloudOptimizer
        if len(imgs) > 2
        else GlobalAlignerMode.PairViewer
    )
    scene = global_aligner(output, device=device, mode=mode)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init="mst", niter=niter, schedule=schedule, lr=lr
        )

    outfile = get_3d_model_from_scene(
        outdir,
        scene,
        min_conf_thr,
        as_pointcloud,
        mask_sky,
        clean_depth,
        transparent_cams,
        cam_size,
    )

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = plt.get_cmap("jet")
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs
