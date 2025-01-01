"""
Codes are copied from https://github.com/hbb1/2d-gaussian-splatting
"""

from typing import Iterable
import torch
import numpy as np
import trimesh
import open3d as o3d
from skimage import measure
from tqdm.auto import tqdm


class GS2DMeshUtils:
    @classmethod
    @torch.no_grad()
    def render_views(cls, model, renderer, cameras: Iterable, bg_color: torch.Tensor):
        rgbmaps = []
        depthmaps = []

        for i, viewpoint_cam in tqdm(enumerate(cameras), total=len(cameras), desc="Rendering RGB and depth maps"):
            render_pkg = renderer(viewpoint_cam, model, bg_color)
            rgb = render_pkg['render']
            # alpha = render_pkg['rend_alpha']
            # normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth = render_pkg['surf_depth']
            # depth_normal = render_pkg['surf_normal']
            rgbmaps.append(rgb.cpu())
            depthmaps.append(depth.cpu())
            # self.alphamaps.append(alpha.cpu())
            # self.normals.append(normal.cpu())
            # self.depth_normals.append(depth_normal.cpu())

        return rgbmaps, depthmaps

    @classmethod
    @torch.no_grad()
    def estimate_bounding_sphere(cls, cameras: Iterable):
        """
        Estimate the bounding sphere given camera pose
        """
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_to_camera.T).cpu().numpy())) for cam in cameras])
        poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
        center = (cls.focus_point_fn(poses))
        radius = np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min()
        center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {radius:.2f}")
        print(f"Use at least {2.0 * radius:.2f} for depth_trunc")

        return center, radius

    @staticmethod
    @torch.no_grad()
    def focus_point_fn(poses: np.ndarray) -> np.ndarray:
        """Calculate nearest point to all focal axes in poses."""
        directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
        m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
        mt_m = np.transpose(m, [0, 2, 1]) @ m
        focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
        return focus_pt

    @staticmethod
    def marching_cubes_with_contraction(
        sdf,
        resolution=512,
        bounding_box_min=(-1.0, -1.0, -1.0),
        bounding_box_max=(1.0, 1.0, 1.0),
        return_mesh=False,
        level=0,
        simplify_mesh=True,
        inv_contraction=None,
        max_range=32.0,
    ):
        assert resolution % 512 == 0

        resN = resolution
        cropN = 512
        level = 0
        N = resN // cropN

        grid_min = bounding_box_min
        grid_max = bounding_box_max
        xs = np.linspace(grid_min[0], grid_max[0], N + 1)
        ys = np.linspace(grid_min[1], grid_max[1], N + 1)
        zs = np.linspace(grid_min[2], grid_max[2], N + 1)

        meshes = []
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    print(i, j, k)
                    x_min, x_max = xs[i], xs[i + 1]
                    y_min, y_max = ys[j], ys[j + 1]
                    z_min, z_max = zs[k], zs[k + 1]

                    x = torch.linspace(x_min, x_max, cropN).cuda()
                    y = torch.linspace(y_min, y_max, cropN).cuda()
                    z = torch.linspace(z_min, z_max, cropN).cuda()

                    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
                    points = torch.tensor(torch.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

                    @torch.no_grad()
                    def evaluate(points):
                        z = []
                        for _, pnts in enumerate(torch.split(points, 256**3, dim=0)):
                            z.append(sdf(pnts))
                        z = torch.cat(z, axis=0)
                        return z

                    # construct point pyramids
                    points = points.reshape(cropN, cropN, cropN, 3)
                    points = points.reshape(-1, 3)
                    pts_sdf = evaluate(points.contiguous())
                    z = pts_sdf.detach().cpu().numpy()
                    if not (np.min(z) > level or np.max(z) < level):
                        z = z.astype(np.float32)
                        verts, faces, normals, _ = measure.marching_cubes(
                            volume=z.reshape(cropN, cropN, cropN),
                            level=level,
                            spacing=(
                                (x_max - x_min) / (cropN - 1),
                                (y_max - y_min) / (cropN - 1),
                                (z_max - z_min) / (cropN - 1),
                            ),
                        )
                        verts = verts + np.array([x_min, y_min, z_min])
                        meshcrop = trimesh.Trimesh(verts, faces, normals)
                        meshes.append(meshcrop)

                    print("finished one block")

        combined = trimesh.util.concatenate(meshes)
        combined.merge_vertices(digits_vertex=6)

        # inverse contraction and clipping the points range
        if inv_contraction is not None:
            combined.vertices = inv_contraction(torch.from_numpy(combined.vertices).float().cuda()).cpu().numpy()
            combined.vertices = np.clip(combined.vertices, -max_range, max_range)

        return combined

    @classmethod
    @torch.no_grad()
    def extract_mesh_unbounded(cls, maps, bound, cameras: Iterable, model, resolution: int = 1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2 - mag) * (y / mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1) @ viewpoint_cam.full_projection
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1.) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3, -1).T
            sdf = (sampled_depth - z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1 / (2 - torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:, 0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:, 0])
            for i, viewpoint_cam in tqdm(enumerate(cameras), total=len(cameras), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                                                           depthmap=depthmaps[i],
                                                           rgbmap=rgbmaps[i],
                                                           viewpoint_cam=cameras[i],
                                                           )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:, None] + rgb[mask_proj]) / wp[:, None]
                # update weight
                weights[mask_proj] = wp

            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        rgbmaps, depthmaps = maps
        center, radius = bound

        def normalize(x): return (x - center) / radius
        def unnormalize(x): return (x * radius) + center
        def inv_contraction(x): return unnormalize(uncontract(x))

        N = resolution
        voxel_size = (radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        def sdf_function(x): return compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        R = contract(normalize(model.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R + 0.01, 1.9)

        mesh = cls.marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )

        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @staticmethod
    def to_cam_open3d(viewpoint_stack):
        camera_traj = []
        for i, viewpoint_cam in enumerate(viewpoint_stack):
            W = viewpoint_cam.width
            H = viewpoint_cam.height
            ndc2pix = torch.tensor([
                [W / 2, 0, 0, (W - 1) / 2],
                [0, H / 2, 0, (H - 1) / 2],
                [0, 0, 0, 1]]).float().cuda().T
            intrins = (viewpoint_cam.projection @ ndc2pix)[:3, :3].T
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=viewpoint_cam.width,
                height=viewpoint_cam.height,
                cx=intrins[0, 2].item(),
                cy=intrins[1, 2].item(),
                fx=intrins[0, 0].item(),
                fy=intrins[1, 1].item()
            )

            extrinsic = np.asarray((viewpoint_cam.world_to_camera.T).cpu().numpy())
            camera = o3d.camera.PinholeCameraParameters()
            camera.extrinsic = extrinsic
            camera.intrinsic = intrinsic
            camera_traj.append(camera)

        return camera_traj

    @classmethod
    @torch.no_grad()
    def extract_mesh_bounded(
        cls,
        maps,
        cameras,
        voxel_size=0.004,
        sdf_trunc=0.02,
        depth_trunc=3,
        mask_backgrond=False,
    ):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        rgbmaps, depthmaps = maps

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        with tqdm(enumerate(cls.to_cam_open3d(cameras)), total=len(cameras), desc="TSDF integration progress") as t:
            for i, cam_o3d in t:
                rgb = rgbmaps[i]
                depth = depthmaps[i]

                # if we have mask provided, use it
                assert mask_backgrond is False
                # if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                #     depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

                # make open3d rgbd
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                    o3d.geometry.Image(np.asarray(depth.permute(1, 2, 0).cpu().numpy(), order="C")),
                    depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
                    depth_scale=1.0
                )

                volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh


def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} cluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0
