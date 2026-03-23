import array
import os
import sys
from typing import Dict, List, Optional, Tuple

import Imath
import numpy as np
import OpenEXR
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from utils.graphics_utils import load_and_transform_diffusion_renderer_image
from utils.training.normal_utils import get_camera_to_world_rotation_matrix, load_normal_image


def _configure_optix_environment():
    optix_build_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Optix", "build"
    )
    if os.path.exists(optix_build_path):
        if optix_build_path not in sys.path:
            sys.path.insert(0, optix_build_path)
        os.environ["PYTHONPATH"] = optix_build_path + os.pathsep + os.environ.get("PYTHONPATH", "")

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cuda_lib_path = os.path.join(cuda_home, "lib64")
    if os.path.exists(cuda_lib_path):
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if optix_build_path not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = cuda_lib_path + os.pathsep + optix_build_path + os.pathsep + current_ld_path

    os.environ.setdefault("CC", "gcc")
    os.environ.setdefault("CXX", "g++")
    os.environ.setdefault("CUDAHOSTCXX", "g++")


_configure_optix_environment()
import SampleRenderer  # noqa: E402


class SimpleIntersectionTracer:
    """Intersection tracer for Gaussian Splatting using OptiX."""

    def __init__(self):
        self.gaussians_data = None
        self.image_width = None
        self.image_height = None
        self.rotation_matrices = None
        self.bounding_boxes = None
        self.model = None
        self.renderer = None

    def load_gaussians(self, centers, scales, rotations, opacities):
        self.gaussians_data = {
            "centers": centers,
            "scales": np.clip(scales / np.sqrt(3), 0.001, 1000),
            "rotations": rotations,
            "opacities": opacities,
        }
        self.calc_rots_and_aabb(centers, scales, rotations, None, None)
        self.create_model_with_bboxes(centers.shape[0])

    def create_model_with_bboxes(self, num_gaussians):
        self.model = SampleRenderer.Model()
        self.model.python_side_assign(self.bounding_boxes, num_gaussians)

    def set_dimensions_from_scene(self, scene):
        if scene is not None:
            training_cameras = scene.getTrainCameras()
            if len(training_cameras) > 0:
                first_camera = training_cameras[0]
                self.image_width = first_camera.image_width
                self.image_height = first_camera.image_height
                print(f"✓ Set tracer dimensions from scene: {self.image_width}x{self.image_height}")
                return True
        return False

    def create_renderer_and_update_parameters(self, num_gaussians):
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model not created. Call create_model_with_bboxes first.")
        self.renderer = SampleRenderer.SampleRenderer(self.model)
        self.update_renderer_parameters(num_gaussians)

    def update_renderer_parameters(self, num_gaussians):
        centers_gpu = torch.from_numpy(self.gaussians_data["centers"]).cuda().float()
        scales_gpu = torch.from_numpy(self.gaussians_data["scales"]).cuda().float()
        rotation_mats_gpu = torch.from_numpy(self.rotation_matrices).cuda().float()
        densities_gpu = torch.from_numpy(self.gaussians_data["opacities"]).cuda().float()
        self.renderer.updateParameters(
            centers_gpu,
            scales_gpu,
            rotation_mats_gpu,
            densities_gpu,
            num_gaussians,
            True,
        )

    def setCameraFromTorch(self, camera_matrix, scene_camera=None):
        try:
            with torch.no_grad():
                if not isinstance(camera_matrix, torch.Tensor):
                    c2w = camera_matrix
                else:
                    c2w = camera_matrix.detach().cpu().contiguous().numpy()

                pos = c2w[:3, 3]
                forwards = c2w[:3, 2]
                up = c2w[:3, 1]
                camera = SampleRenderer.Camera(
                    SampleRenderer.vec3f(float(pos[0]), float(pos[1]), float(pos[2])),
                    SampleRenderer.vec3f(float(forwards[0]), float(forwards[1]), float(forwards[2])),
                    SampleRenderer.vec3f(float(up[0]), float(up[1]), float(up[2])),
                )

                if scene_camera is not None:
                    fx = _to_float(getattr(scene_camera, "fx", None))
                    fy = _to_float(getattr(scene_camera, "fy", None))
                    cx = _to_float(getattr(scene_camera, "cx", 0.0)) if getattr(scene_camera, "cx", None) is not None else 0.0
                    cy = _to_float(getattr(scene_camera, "cy", 0.0)) if getattr(scene_camera, "cy", None) is not None else 0.0

                    if (fx is None or fy is None) and hasattr(scene_camera, "FoVx") and hasattr(scene_camera, "FoVy"):
                        from utils.graphics_utils import fov2focal

                        fovx = _to_float(getattr(scene_camera, "FoVx", None))
                        fovy = _to_float(getattr(scene_camera, "FoVy", None))
                        img_w = _to_int(getattr(scene_camera, "image_width", None))
                        img_h = _to_int(getattr(scene_camera, "image_height", None))
                        if fx is None and fovx is not None and img_w and img_w > 0:
                            fx = float(fov2focal(fovx, img_w))
                        if fy is None and fovy is not None and img_h and img_h > 0:
                            fy = float(fov2focal(fovy, img_h))

                    if fx is not None and fy is not None and fx > 0 and fy > 0:
                        self.renderer.setCameraFromIntrinsics(camera, float(fx), float(fy), float(cx), float(cy))
                    else:
                        self.renderer.setCamera(camera)
                else:
                    self.renderer.setCamera(camera)
        except Exception as e:
            print("\nError in setCameraFromTorch:")
            print(f"Error message: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    def calc_rots_and_aabb(self, centers, scales, rotations, rotationMats, boundingBoxes):
        centers_gpu = torch.from_numpy(centers).cuda().float()
        scales_gpu = torch.from_numpy(scales).cuda().float()
        rotations_gpu = torch.from_numpy(rotations).cuda().float()

        num_gaussians = centers.shape[0]
        rotationMats_gpu = torch.zeros((num_gaussians, 9), device="cuda", dtype=torch.float32)
        boundingBoxes_gpu = torch.zeros((num_gaussians, 2, 3), device="cuda", dtype=torch.float32)

        SampleRenderer.calc_rots_and_aabb(centers_gpu, scales_gpu, rotations_gpu, rotationMats_gpu, boundingBoxes_gpu)
        self.rotation_matrices = rotationMats_gpu.detach().cpu().numpy()
        self.bounding_boxes = boundingBoxes_gpu

    def create_renderer(self):
        if not hasattr(self, "gaussians_data") or self.gaussians_data is None:
            raise ValueError("No Gaussian data loaded. Call load_gaussians() first.")
        num_gaussians = self.gaussians_data["centers"].shape[0]
        self.create_renderer_and_update_parameters(num_gaussians)

    def close(self):
        try:
            if hasattr(self, "renderer") and self.renderer is not None:
                for method_name in ["free", "destroy", "shutdown", "release", "__del__"]:
                    try:
                        method = getattr(self.renderer, method_name, None)
                        if callable(method):
                            method()
                    except Exception:
                        pass
        finally:
            self.renderer = None
            self.model = None
            self.bounding_boxes = None
            self.rotation_matrices = None
            self.gaussians_data = None
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def has_renderer(self):
        return hasattr(self, "renderer") and self.renderer is not None

    def get_intersection_data(self):
        if not self.has_renderer():
            raise ValueError("Renderer not initialized. Call create_renderer() first.")

        width = self.image_width
        height = self.image_height
        max_intersections = 150
        try:
            intersection_ids_cpu = np.zeros(height * width * max_intersections, dtype=np.uint32)
            num_intersections_cpu = np.zeros(height * width, dtype=np.uint32)
            self.renderer.downloadIntersectionData(intersection_ids_cpu, height * width * max_intersections)
            self.renderer.downloadNumIntersectionsData(num_intersections_cpu, height * width)
            intersection_ids = intersection_ids_cpu.reshape(height, width, max_intersections)
            num_intersections = num_intersections_cpu.reshape(height, width)
        except Exception as e:
            print(f"Error downloading data from GPU: {e}")
            print(f"Image dimensions: {width}x{height}, max_intersections: {max_intersections}")
            raise
        return intersection_ids, num_intersections

    def assign_colors_to_gaussians_gpu(self, pixel_colors, intersection_ids, num_intersections):
        if self.gaussians_data is None:
            raise ValueError("No Gaussian data loaded. Call load_gaussians() first.")

        num_gaussians = self.gaussians_data["centers"].shape[0]
        max_intersections = intersection_ids.shape[2]
        max_hits_per_gaussian = 128
        gaussian_colors = torch.zeros((num_gaussians, 3), device="cuda", dtype=torch.float32)
        gaussian_hit_counts = torch.zeros(num_gaussians, device="cuda", dtype=torch.int32)
        SampleRenderer.processIntersectionMedianColors(
            intersection_ids,
            num_intersections,
            pixel_colors,
            gaussian_colors,
            gaussian_hit_counts,
            max_intersections,
            num_gaussians,
            max_hits_per_gaussian,
        )
        return gaussian_colors, gaussian_hit_counts


def _to_float(v):
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return float(v.item())
    if isinstance(v, (np.ndarray, np.generic)):
        return float(v)
    return float(v)


def _to_int(v):
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return int(v.item())
    return int(v)


def extract_gaussian_parameters(gaussians):
    with torch.no_grad():
        centers = gaussians.get_xyz.detach().cpu().numpy()
        scales = gaussians.get_scaling.detach().cpu().numpy()
        rotations = gaussians.get_rotation.detach().cpu().numpy()
        opacities = gaussians.get_opacity.detach().cpu().numpy()
        return centers, scales, rotations, opacities


def create_camera_matrix_from_scene_camera(scene, camera_index=0):
    if scene is None:
        return None
    cameras = scene.getTrainCameras()
    if len(cameras) == 0:
        return None
    camera = cameras[camera_index] if camera_index < len(cameras) else cameras[0]
    return camera.c2w.detach().cpu().numpy()


def _get_number_parts(image_name: str) -> List[str]:
    number_part = os.path.splitext(image_name)[0]
    number_parts = [number_part]
    if number_part.startswith("r_"):
        number_parts.append("r_r_" + number_part[2:])
    elif number_part.startswith("r_r_"):
        number_parts.append("r_" + number_part[4:])
    return number_parts


def _resolve_first_existing_path(source_path: str, folder: str, filenames: List[str]) -> Tuple[Optional[str], List[str]]:
    folder = folder[1:] if folder.startswith("/") else folder
    attempted = []
    for filename in filenames:
        path = os.path.join(source_path, folder, filename)
        attempted.append(path)
        if os.path.exists(path):
            return path, attempted
    return None, attempted


def _load_camera_original_image(camera, camera_name: str):
    if (not hasattr(camera, "original_image")) or (camera.original_image is None):
        assert hasattr(camera, "image_path") and camera.image_path is not None, "camera.image_path missing"
        img = torchvision.io.read_image(camera.image_path).float()
        if img.max() > 1.0:
            img = img / 255.0
        camera.original_image = img.to("cuda")
        if img.shape[0] == 4:
            camera.image_mask = img[3:4].to("cuda").clamp(0.0, 1.0)
    return camera.original_image


def _load_projected_normal_tensor(normal_gt_path: str, args, camera, device):
    if getattr(args, "normals_are_world_space", False):
        if normal_gt_path.endswith(".exr"):
            exr_file = OpenEXR.InputFile(normal_gt_path)
            dw = exr_file.header()["dataWindow"]
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            float_type = Imath.PixelType(Imath.PixelType.FLOAT)
            r = np.array(array.array("f", exr_file.channel("R", float_type))).reshape(size[1], size[0])
            g = np.array(array.array("f", exr_file.channel("G", float_type))).reshape(size[1], size[0])
            b = np.array(array.array("f", exr_file.channel("B", float_type))).reshape(size[1], size[0])
            exr_file.close()
            normal_np = np.stack([r, g, b], axis=0).astype(np.float32)
            normal_gt = torch.from_numpy(normal_np)
        else:
            normal_gt = torchvision.io.read_image(normal_gt_path).float()
            if normal_gt.shape[0] >= 4:
                normal_gt = normal_gt[:3]
            if normal_gt.max() > 1.0:
                normal_gt = normal_gt / 255.0
        if getattr(args, "normals_range", "0_1") == "0_1":
            normal_gt = normal_gt * 2.0 - 1.0
        normal_gt = F.normalize(normal_gt, dim=0, eps=1e-6).to(device)
        return normal_gt.permute(1, 2, 0)

    normal_gt = load_normal_image(normal_gt_path).to(device)
    camera_rotation_matrix = get_camera_to_world_rotation_matrix(camera)
    camera_rotation_tensor = torch.from_numpy(camera_rotation_matrix).float().to(device)
    h, w = normal_gt.shape[1], normal_gt.shape[2]
    normal_flat = normal_gt.permute(1, 2, 0).reshape(-1, 3)
    world_normals_flat = torch.matmul(normal_flat, camera_rotation_tensor)
    world_normals_flat = F.normalize(world_normals_flat, dim=-1, eps=1e-6)
    return world_normals_flat.reshape(h, w, 3)


def _load_modality_images(camera, args, dataset):
    device = camera.original_image.device
    number_parts = _get_number_parts(camera.image_name)

    base_color_gpu = None
    if args.base_color_folder is not None:
        filenames = []
        for num_part in number_parts:
            if num_part.startswith("r_r_"):
                filenames.append(f"{num_part}_albedo.png")
            else:
                filenames.extend([f"{num_part}_basecolor0001.png", f"{num_part}_basecolor.png", f"{num_part}_albedo.png"])
        path, attempted = _resolve_first_existing_path(dataset.source_path, args.base_color_folder, filenames)
        if path is not None:
            base_color_gt = load_and_transform_diffusion_renderer_image(path, modality="basecolor", device=device)
            base_color_gpu = base_color_gt.permute(1, 2, 0)
        else:
            print(f"[WARNING] Base Color - Image not found for camera {camera.image_name}. Searched in these paths:")
            for p in attempted:
                print(f"[WARNING] Base Color -   {p}")

    roughness_gpu = None
    if args.roughness_folder is not None:
        filenames = []
        for num_part in number_parts:
            if num_part.startswith("r_r_"):
                filenames.extend([f"{num_part}_roughness.png", f"{num_part}_metallicroughness.png"])
            else:
                filenames.extend(
                    [
                        f"{num_part}_roughness0001.png",
                        f"{num_part}_metallicroughness0001.png",
                        f"{num_part}_roughness.png",
                        f"{num_part}_metallicroughness.png",
                    ]
                )
        path, attempted = _resolve_first_existing_path(dataset.source_path, args.roughness_folder, filenames)
        if path is not None:
            roughness_gt = load_and_transform_diffusion_renderer_image(path, modality="roughness", device=device)
            roughness_gt_green = roughness_gt[1:2] if roughness_gt.shape[0] >= 3 else roughness_gt
            roughness_gpu = roughness_gt_green.repeat(3, 1, 1).permute(1, 2, 0)
        else:
            print(f"[WARNING] Roughness - Image not found for camera {camera.image_name}. Searched in these paths:")
            for p in attempted:
                print(f"[WARNING] Roughness -   {p}")

    metallic_gpu = None
    if args.metallic_folder is not None:
        filenames = []
        for num_part in number_parts:
            if num_part.startswith("r_r_"):
                filenames.extend([f"{num_part}_metallic.png", f"{num_part}_metallicroughness.png"])
            else:
                filenames.extend(
                    [
                        f"{num_part}_metallic0001.png",
                        f"{num_part}_metallicroughness0001.png",
                        f"{num_part}_metallic.png",
                        f"{num_part}_metallicroughness.png",
                    ]
                )
        path, attempted = _resolve_first_existing_path(dataset.source_path, args.metallic_folder, filenames)
        if path is not None:
            metallic_gt = load_and_transform_diffusion_renderer_image(path, modality="metallic", device=device)
            metallic_gt_channel = metallic_gt[0:1] if metallic_gt.shape[0] >= 3 else metallic_gt
            metallic_gpu = metallic_gt_channel.repeat(3, 1, 1).permute(1, 2, 0)
        else:
            print(f"[WARNING] Metallic - Image not found for camera {camera.image_name}. Searched in these paths:")
            for p in attempted:
                print(f"[WARNING] Metallic -   {p}")

    normal_gpu = None
    if args.normals_folder is not None:
        filenames = []
        for num_part in number_parts:
            if num_part.startswith("r_r_"):
                filenames.extend([f"{num_part}_normal.png", f"{num_part}_normal.exr"])
            else:
                filenames.extend(
                    [
                        f"{num_part}_normal0001.png",
                        f"{num_part}_normal.png",
                        f"{num_part}_normal0001.exr",
                        f"{num_part}_normal.exr",
                        f"{num_part}_normalcamera0001.exr",
                    ]
                )
        path, attempted = _resolve_first_existing_path(dataset.source_path, args.normals_folder, filenames)
        if path is not None:
            try:
                normal_gpu = _load_projected_normal_tensor(path, args, camera, device)
            except Exception as e:
                print(f"Error loading/transforming normal image {path}: {e}")
        else:
            print(f"[WARNING] Normal - Image not found for camera {camera.image_name}. Searched in these paths:")
            for p in attempted:
                print(f"[WARNING] Normal -   {p}")

    return base_color_gpu, roughness_gpu, metallic_gpu, normal_gpu


def perform_intersection_tracing_for_all_training_images(
    gaussians, scene, model_path, width=None, height=None, args=None, dataset=None
):
    print("\n=== Starting Intersection Tracing for All Training Images ===")
    centers, scales, rotations, opacities = extract_gaussian_parameters(gaussians)
    num_gaussians = centers.shape[0]
    print(f"Processing {num_gaussians} Gaussians")

    tracer = SimpleIntersectionTracer()
    if not tracer.set_dimensions_from_scene(scene):
        if width is None or height is None:
            width = 512
            height = 512
            print(f"Warning: No training cameras found, using default dimensions: {width}x{height}")
        tracer.image_width = width
        tracer.image_height = height

    print("Loading Gaussians into intersection tracer...")
    tracer.load_gaussians(centers, scales, rotations, opacities)
    tracer.create_renderer()

    training_cameras = scene.getTrainCameras()
    print(f"Found {len(training_cameras)} training cameras")
    all_gaussian_base_colors = torch.zeros((num_gaussians, len(training_cameras), 3), device="cuda", dtype=torch.float32)
    all_gaussian_roughness = torch.zeros((num_gaussians, len(training_cameras), 1), device="cuda", dtype=torch.float32)
    all_gaussian_metallic = torch.zeros((num_gaussians, len(training_cameras), 1), device="cuda", dtype=torch.float32)
    all_gaussian_normals = torch.zeros((num_gaussians, len(training_cameras), 3), device="cuda", dtype=torch.float32)
    gaussian_hit_mask = torch.zeros((num_gaussians, len(training_cameras)), device="cuda", dtype=torch.bool)

    intersection_data_dir = os.path.join(model_path, "intersection_data")
    os.makedirs(intersection_data_dir, exist_ok=True)

    for camera_idx, camera in enumerate(tqdm(training_cameras, desc="Processing training cameras")):
        camera_name = camera.image_name
        try:
            camera_matrix = create_camera_matrix_from_scene_camera(scene, camera_idx)
            if camera_matrix is None:
                print(f"Warning: Could not create camera matrix for {camera_name}, skipping...")
                continue

            _load_camera_original_image(camera, camera_name)
            tracer.setCameraFromTorch(camera_matrix, scene_camera=camera)
            tracer.renderer.resize(SampleRenderer.vec2i(width, height))
            tracer.renderer.render()
            intersection_ids, num_intersections = tracer.get_intersection_data()

            base_color_gpu, roughness_gpu, metallic_gpu, normal_gpu = _load_modality_images(camera, args, dataset)
            if base_color_gpu is not None:
                projection_image = base_color_gpu
            else:
                gt_image = camera.original_image.cuda()
                if gt_image.shape[0] == 4:
                    gt_image = gt_image[:3]
                projection_image = gt_image.permute(1, 2, 0)

            intersection_ids_gpu = torch.from_numpy(intersection_ids).cuda().to(torch.uint32)
            num_intersections_gpu = torch.from_numpy(num_intersections).cuda().to(torch.uint32)
            gaussian_base_colors, gaussian_hit_counts = tracer.assign_colors_to_gaussians_gpu(
                projection_image, intersection_ids_gpu, num_intersections_gpu
            )
            all_gaussian_base_colors[:, camera_idx, :] = gaussian_base_colors
            hit_mask = gaussian_hit_counts > 0
            gaussian_hit_mask[:, camera_idx] = hit_mask

            if roughness_gpu is not None:
                gaussian_roughness, _ = tracer.assign_colors_to_gaussians_gpu(
                    roughness_gpu, intersection_ids_gpu, num_intersections_gpu
                )
                all_gaussian_roughness[:, camera_idx, :] = gaussian_roughness[:, :1]
            if metallic_gpu is not None:
                gaussian_metallic, _ = tracer.assign_colors_to_gaussians_gpu(
                    metallic_gpu, intersection_ids_gpu, num_intersections_gpu
                )
                all_gaussian_metallic[:, camera_idx, :] = gaussian_metallic[:, :1]
            if normal_gpu is not None:
                gaussian_normals, _ = tracer.assign_colors_to_gaussians_gpu(
                    normal_gpu, intersection_ids_gpu, num_intersections_gpu
                )
                all_gaussian_normals[:, camera_idx, :] = gaussian_normals

            try:
                del intersection_ids_gpu, num_intersections_gpu
                del projection_image, base_color_gpu, roughness_gpu, metallic_gpu, normal_gpu
                torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as e:
            print(f"  ✗ Error processing camera {camera_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    try:
        tracer.close()
    except Exception:
        pass
    return all_gaussian_base_colors, all_gaussian_roughness, all_gaussian_metallic, all_gaussian_normals, gaussian_hit_mask


def load_intersection_data(model_path):
    intersection_data_dir = os.path.join(model_path, "intersection_data")
    if not os.path.exists(intersection_data_dir):
        print(f"Intersection data directory not found: {intersection_data_dir}")
        return None

    summary_file = os.path.join(intersection_data_dir, "intersection_summary.json")
    if os.path.exists(summary_file):
        import json

        with open(summary_file, "r") as f:
            summary = json.load(f)
        print(f"Loaded intersection summary: {summary}")
    else:
        print("Intersection summary file not found")
        return None

    gaussian_intersection_data: Dict[int, Dict] = {}
    num_gaussians = summary["num_gaussians"]
    print(f"Loading intersection data for {num_gaussians} Gaussians...")
    for gaussian_id in tqdm(range(num_gaussians), desc="Loading Gaussian data"):
        gaussian_file = os.path.join(intersection_data_dir, f"gaussian_{gaussian_id:06d}_data.npz")
        if os.path.exists(gaussian_file):
            data = np.load(gaussian_file)
            gaussian_intersection_data[gaussian_id] = {
                "camera_names": data["camera_names"].tolist(),
                "colors": data["colors"],
                "hit_counts": data["hit_counts"],
                "total_intersections": data["total_intersections"],
            }

    print(f"Loaded intersection data for {len(gaussian_intersection_data)} Gaussians")
    return gaussian_intersection_data


def get_gaussian_intersection_info(gaussian_id, intersection_data, camera_name=None):
    if intersection_data is None or gaussian_id not in intersection_data:
        return None
    gaussian_data = intersection_data[gaussian_id]
    if camera_name is not None:
        if camera_name in gaussian_data["camera_names"]:
            idx = gaussian_data["camera_names"].index(camera_name)
            return {
                "camera_name": camera_name,
                "colors": gaussian_data["colors"][idx],
                "hit_count": gaussian_data["hit_counts"][idx],
                "total_intersections": gaussian_data["total_intersections"][idx],
            }
        return None
    return gaussian_data


def get_gaussians_hit_by_camera(intersection_data, camera_name):
    if intersection_data is None:
        return []
    hit_gaussians = []
    for gaussian_id, gaussian_data in intersection_data.items():
        if camera_name in gaussian_data["camera_names"]:
            hit_gaussians.append(gaussian_id)
    return hit_gaussians
