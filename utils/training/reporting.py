import array
import os

import Imath
import OpenEXR
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from lpipsPyTorch import lpips
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

from utils.image_utils import psnr, visualize_depth
from utils.loss_utils import ssim
from utils.graphics_utils import rgb_to_srgb
from utils.training.normal_utils import get_camera_to_world_rotation_matrix


def _load_normal_gt_as_rgb(viewpoint, args, source_path_fallback):
    original_basename = viewpoint.image_name
    number_part = os.path.splitext(original_basename)[0]
    number_parts = [number_part]
    if number_part.startswith("r_"):
        number_parts.append("r_r_" + number_part[2:])
    elif number_part.startswith("r_r_"):
        number_parts.append("r_" + number_part[4:])

    normal_filename_options = []
    for num_part in number_parts:
        normal_filename_options.extend(
            [
                f"{num_part}_normalcamera0001.exr",
                f"{num_part}_normal0001.exr",
                f"{num_part}_normal.exr",
                f"{num_part}_normal.png",
            ]
        )

    normal_gt_path = None
    normal_folder = args.normal_gt_folder
    if normal_folder.startswith("/"):
        normal_folder = normal_folder[1:]

    for fname in normal_filename_options:
        p = os.path.join(source_path_fallback, normal_folder, fname)
        if os.path.exists(p):
            normal_gt_path = p
            break

    if normal_gt_path is None:
        return viewpoint.original_image.cuda()

    if normal_gt_path.endswith(".exr"):
        file = OpenEXR.InputFile(normal_gt_path)
        dw = file.header()["dataWindow"]
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        float_type = Imath.PixelType(Imath.PixelType.FLOAT)
        r = np.array(array.array("f", file.channel("R", float_type))).reshape(size[1], size[0])
        g = np.array(array.array("f", file.channel("G", float_type))).reshape(size[1], size[0])
        b = np.array(array.array("f", file.channel("B", float_type))).reshape(size[1], size[0])
        file.close()
        normal_gt = torch.from_numpy(np.stack([r, g, b], axis=0)).float()
    else:
        normal_gt = torchvision.io.read_image(normal_gt_path).float()
        if normal_gt.shape[0] >= 4:
            normal_gt = normal_gt[:3]
        if normal_gt.max() > 1.0:
            normal_gt = normal_gt / 255.0

    device = viewpoint.original_image.device if hasattr(viewpoint, "original_image") else "cuda"
    normal_gt = normal_gt.to(device)

    if getattr(args, "normal_gt_is_camera_space", False):
        camera_rotation_matrix = get_camera_to_world_rotation_matrix(viewpoint)
        camera_rotation_tensor = torch.from_numpy(camera_rotation_matrix).float().to(device)
        h_n, w_n = normal_gt.shape[1], normal_gt.shape[2]
        normal_flat = normal_gt.permute(1, 2, 0).reshape(-1, 3)
        world_flat = torch.matmul(normal_flat, camera_rotation_tensor)
        world_flat = F.normalize(world_flat, dim=-1, eps=1e-6)
        normal_gt = world_flat.reshape(h_n, w_n, 3).permute(2, 0, 1)
    else:
        normal_gt = F.normalize(normal_gt, dim=0, eps=1e-6)

    if getattr(args, "normal_gt_range", "-1_1") == "-1_1":
        return torch.clamp(normal_gt * 0.5 + 0.5, 0.0, 1.0)
    return torch.clamp(normal_gt, 0.0, 1.0)


def training_report(tb_writer, iteration, tb_dict, scene, renderFunc, pipe, bg_color, args, scaling_modifier=1.0, override_color=None, opt=None, is_training=False, **kwargs):
    if tb_writer:
        for key in tb_dict:
            tb_writer.add_scalar(f"train_loss_patches/{key}", tb_dict[key], iteration)

    if iteration % args.test_interval != 0:
        return

    torch.cuda.empty_cache()
    validation_configs = ({"name": "test", "cameras": scene.getTestCameras()}, {"name": "train", "cameras": scene.getTrainCameras()})
    for config in validation_configs:
        if not (config["cameras"] and len(config["cameras"]) > 0):
            continue

        l1_test = 0.0
        psnr_test = 0.0
        psnr_pbr_test = 0.0
        for idx, viewpoint in enumerate(tqdm(config["cameras"], desc="Evaluating " + config["name"], leave=False)):
            render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg_color, scaling_modifier, override_color, opt, is_training, **kwargs)
            image = render_pkg["render"]

            if getattr(args, "use_normals_as_rgb_gt", False) and getattr(args, "normal_gt_folder", None) is not None:
                try:
                    source_path = scene.dataset.source_path if hasattr(scene, "dataset") else args.source_path
                    gt_image = _load_normal_gt_as_rgb(viewpoint, args, source_path)
                except Exception:
                    gt_image = viewpoint.original_image.cuda()
            else:
                gt_image = viewpoint.original_image.cuda()

            if gt_image.shape[0] == 4:
                gt_image = gt_image[:3]
            elif gt_image.shape[0] == 1:
                gt_image = gt_image.repeat(3, 1, 1)

            opacity = torch.clamp(render_pkg["opacity"], 0.0, 1.0)
            depth = render_pkg["depth"]
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            normal = torch.clamp(render_pkg.get("normal", torch.zeros_like(image)) / 2 + 0.5 * opacity, 0.0, 1.0)
            base_color = torch.clamp(render_pkg.get("base_color", torch.zeros_like(image)), 0.0, 1.0)
            roughness = torch.clamp(render_pkg.get("roughness", torch.zeros_like(depth)), 0.0, 1.0)
            metallic = torch.clamp(render_pkg.get("metallic", torch.zeros_like(depth)), 0.0, 1.0)
            image_pbr = render_pkg.get("pbr", torch.zeros_like(image))
            grid = torchvision.utils.make_grid(
                torch.stack(
                    [
                        image,
                        image_pbr,
                        gt_image,
                        opacity.repeat(3, 1, 1),
                        depth.repeat(3, 1, 1),
                        normal,
                        base_color,
                        roughness.repeat(3, 1, 1),
                        metallic.repeat(3, 1, 1),
                    ],
                    dim=0,
                ),
                nrow=3,
            )
            if tb_writer and idx < 2:
                tb_writer.add_images(config["name"] + "_view_{}/render".format(viewpoint.image_name), grid[None], global_step=iteration)

            l1_test += F.l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
            psnr_pbr_test += psnr(image_pbr, gt_image).mean().double()

        psnr_test /= len(config["cameras"])
        psnr_pbr_test /= len(config["cameras"])
        l1_test /= len(config["cameras"])
        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} PSNR_PBR {}".format(iteration, config["name"], l1_test, psnr_test, psnr_pbr_test))
        if tb_writer:
            tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
            tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)
            tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr_pbr", psnr_pbr_test, iteration)
        if iteration == args.iterations:
            with open(os.path.join(args.model_path, config["name"] + "_loss.txt"), "w") as f:
                f.write("L1 {} PSNR {} PSNR_PBR {}".format(l1_test, psnr_test, psnr_pbr_test))

    torch.cuda.empty_cache()


def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, first_iter, iteration, pbr_kwargs, args):
    os.makedirs(os.path.join(args.model_path, "visualize"), exist_ok=True)
    with torch.no_grad():
        if iteration % pipe.save_training_vis_iteration != 0 and iteration != first_iter + 1:
            return
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, opt=opt, is_training=False, dict_params=pbr_kwargs)
        gt_img = viewpoint_cam.original_image.cuda()
        if gt_img.shape[0] == 4:
            gt_img = gt_img[:3]
        elif gt_img.shape[0] == 1:
            gt_img = gt_img.repeat(3, 1, 1)
        visualization_list = [
            render_pkg["render"],
            gt_img,
            visualize_depth(render_pkg["depth"]),
            (render_pkg["depth_var"] / 0.001).clamp_max(1).repeat(3, 1, 1),
            render_pkg["opacity"].repeat(3, 1, 1),
            render_pkg["normal"] * 0.5 + 0.5,
            render_pkg["pseudo_normal"] * 0.5 + 0.5,
        ]
        if gaussians.use_pbr:
            h, w = render_pkg["pbr"].shape[1:]
            env = F.interpolate(render_pkg["env"].permute(0, 3, 1, 2), (h, 2 * w))
            env_0 = env[0, :, :, :w]
            env_1 = env[0, :, :, w:]
            visualization_list.extend(
                [
                    render_pkg["base_color"],
                    render_pkg["roughness"].repeat(3, 1, 1),
                    render_pkg["metallic"].repeat(3, 1, 1),
                    render_pkg["visibility"].repeat(3, 1, 1),
                    render_pkg["diffuse"],
                    render_pkg["specular"],
                    render_pkg["global_lights"],
                    render_pkg["pbr"],
                    rgb_to_srgb(env_0),
                    rgb_to_srgb(env_1),
                ]
            )
        grid = torch.stack(visualization_list, dim=0)
        grid = make_grid(grid, nrow=4)
        reference_height = getattr(viewpoint_cam, "image_height", 512) or 512
        scale = grid.shape[-2] / reference_height
        grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
        save_image(grid, os.path.join(args.model_path, "visualize", f"{iteration:06d}.png"))


def eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, args):
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("Warning: No test cameras found. Skipping evaluation.")
        return

    os.makedirs(os.path.join(args.model_path, "eval", "render"), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, "eval", "gt"), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, "eval", "normal"), exist_ok=True)
    if gaussians.use_pbr:
        for folder in ["base_color", "roughness", "metallic", "lights", "local", "global", "visibility"]:
            os.makedirs(os.path.join(args.model_path, "eval", folder), exist_ok=True)

    progress_bar = tqdm(range(0, len(test_cameras)), desc="Evaluating", initial=0, total=len(test_cameras))
    with torch.no_grad():
        for idx in progress_bar:
            viewpoint = test_cameras[idx]
            results = render_fn(viewpoint, gaussians, pipe, background, opt=opt, is_training=False, dict_params=pbr_kwargs)
            image = results["pbr"] if gaussians.use_pbr else results["render"]
            image = torch.clamp(image, 0.0, 1.0)

            if getattr(args, "use_normals_as_rgb_gt", False) and getattr(args, "normal_gt_folder", None) is not None:
                try:
                    source_path = scene.dataset.source_path if hasattr(scene, "dataset") else args.source_path
                    gt_image = _load_normal_gt_as_rgb(viewpoint, args, source_path)
                except Exception:
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            else:
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += lpips(image, gt_image, net_type="vgg").mean().double()

            save_image(image, os.path.join(args.model_path, "eval", "render", f"{viewpoint.image_name}.png"))
            save_image(gt_image, os.path.join(args.model_path, "eval", "gt", f"{viewpoint.image_name}.png"))
            save_image(results["normal"] * 0.5 + 0.5, os.path.join(args.model_path, "eval", "normal", f"{viewpoint.image_name}.png"))
            if gaussians.use_pbr:
                save_image(results["base_color"], os.path.join(args.model_path, "eval", "base_color", f"{viewpoint.image_name}.png"))
                save_image(results["roughness"], os.path.join(args.model_path, "eval", "roughness", f"{viewpoint.image_name}.png"))
                save_image(results["metallic"], os.path.join(args.model_path, "eval", "metallic", f"{viewpoint.image_name}.png"))
                save_image(results["lights"], os.path.join(args.model_path, "eval", "lights", f"{viewpoint.image_name}.png"))
                save_image(results["local_lights"], os.path.join(args.model_path, "eval", "local", f"{viewpoint.image_name}.png"))
                save_image(results["global_lights"], os.path.join(args.model_path, "eval", "global", f"{viewpoint.image_name}.png"))
                save_image(results["visibility"], os.path.join(args.model_path, "eval", "visibility", f"{viewpoint.image_name}.png"))

    psnr_test /= len(test_cameras)
    ssim_test /= len(test_cameras)
    lpips_test /= len(test_cameras)
    with open(os.path.join(args.model_path, "eval", "eval.txt"), "w") as f:
        f.write(f"psnr: {psnr_test}\n")
        f.write(f"ssim: {ssim_test}\n")
        f.write(f"lpips: {lpips_test}\n")
    print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {}".format(args.iterations, "test", psnr_test, ssim_test, lpips_test))
