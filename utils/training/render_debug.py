import glob
import json
import os
import subprocess

import cv2
import torch
from tqdm import tqdm
from torchvision.utils import save_image

from utils.camera_utils import JSON_to_camera


def render_test_views(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, args):
    print("\n=== Rendering 5 Test Views ===")
    training_cameras = scene.getTrainCameras()
    if len(training_cameras) == 0:
        print("No training cameras found!")
        return

    test_cameras = training_cameras[:5]
    print(f"Rendering {len(test_cameras)} test views...")
    test_renders_dir = os.path.join(args.model_path, "test_renders")
    os.makedirs(test_renders_dir, exist_ok=True)

    with torch.no_grad():
        for idx, camera in enumerate(test_cameras):
            print(f"Rendering camera {idx+1}/{len(test_cameras)}: {camera.image_name}")
            render_pkg = render_fn(camera, gaussians, pipe, background, opt=opt, is_training=False, dict_params=pbr_kwargs)
            rendered_image = render_pkg["render"]
            output_path = os.path.join(test_renders_dir, f"test_render_{idx:02d}_{camera.image_name}.png")
            save_image(rendered_image, output_path)
            print(f"  Saved to: {output_path}")

    print(f"✓ Test renders saved to: {test_renders_dir}")
    print(f"📁 Full path: {os.path.abspath(test_renders_dir)}")


def render_5_test_images(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, args):
    print("\n=== Rendering 5 Test Images (Direct JSON Loading) ===")
    cameras_json_path = os.path.join(args.model_path, "cameras.json")
    if not os.path.exists(cameras_json_path):
        print(f"❌ cameras.json not found at: {cameras_json_path}")
        return

    print("Loading cameras from JSON...")
    with open(cameras_json_path, "r") as f:
        cameras_data = json.load(f)
    print(f"✓ Loaded {len(cameras_data)} cameras from JSON")

    cameras = []
    for cam_data in cameras_data:
        camera = JSON_to_camera(cam_data)
        img_stem = cam_data.get("img_name")
        candidate_dirs = [
            os.path.join(args.source_path, "train", "images"),
            os.path.join(args.source_path, "train", "images_bg"),
            os.path.join(args.source_path, "images"),
            os.path.join(args.source_path, "images_bg"),
        ]
        candidate_exts = [".png", ".jpg", ".jpeg", ".exr"]
        found_path = None
        for d in candidate_dirs:
            for ext in candidate_exts:
                p = os.path.join(d, img_stem + ext)
                if os.path.exists(p):
                    found_path = p
                    break
            if found_path is not None:
                break
        if found_path is None:
            for d in candidate_dirs:
                try:
                    for name in os.listdir(d):
                        if name.startswith(img_stem):
                            found_path = os.path.join(d, name)
                            break
                    if found_path is not None:
                        break
                except Exception:
                    pass
        setattr(camera, "image_path", found_path)
        cameras.append(camera)

    print(f"✓ Converted {len(cameras)} cameras to Camera objects")
    test_indices = [0, 25, 50]
    test_cameras = [cameras[idx] for idx in test_indices if idx < len(cameras)]
    if len(test_cameras) == 0:
        print("No cameras available for testing!")
        return

    print(f"Rendering {len(test_cameras)} test views...")
    test_renders_dir = os.path.join(args.model_path, "test_renders_direct_json")
    os.makedirs(test_renders_dir, exist_ok=True)

    with torch.no_grad():
        for idx, camera in enumerate(test_cameras):
            print(f"Rendering camera {idx+1}/{len(test_cameras)}: {camera.image_name}")
            render_pkg = render_fn(camera, gaussians, pipe, background, opt=opt, is_training=False, dict_params=pbr_kwargs)
            rendered_image = render_pkg["render"]
            output_path = os.path.join(test_renders_dir, f"test_render_{idx:02d}_{camera.image_name}.png")
            save_image(rendered_image, output_path)
            print(f"  Saved to: {output_path}")
            torch.cuda.empty_cache()

    print(f"✓ Test renders saved to: {test_renders_dir}")
    print(f"📁 Full path: {os.path.abspath(test_renders_dir)}")
    print(f"📊 Rendered {len(test_cameras)} views")


def render_timelapse_frame(timelapse_camera, gaussians, render_fn, pipe, background, opt, pbr_kwargs, iteration, timelapse_dir, is_pbr=False):
    os.makedirs(timelapse_dir, exist_ok=True)
    with torch.no_grad():
        render_pkg = render_fn(timelapse_camera, gaussians, pipe, background, opt=opt, is_training=False, dict_params=pbr_kwargs)
        rendered_image = render_pkg["pbr"] if is_pbr else render_pkg["render"]
        output_path = os.path.join(timelapse_dir, f"timelapse_{iteration:06d}.png")
        save_image(rendered_image, output_path)
        gt_image = timelapse_camera.original_image.cuda()
        gt_path = os.path.join(timelapse_dir, f"gt_{iteration:06d}.png")
        save_image(gt_image, gt_path)
        print(f"📸 Timelapse frame saved: {output_path}")


def create_timelapse_video(timelapse_dir, output_video_path, fps=10):
    try:
        frame_pattern = os.path.join(timelapse_dir, "timelapse_*.png")
        frame_files = sorted(glob.glob(frame_pattern))
        if not frame_files:
            print(f"No timelapse frames found in {timelapse_dir}")
            return False

        print(f"Creating timelapse video from {len(frame_files)} frames...")
        try:
            input_pattern = os.path.join(timelapse_dir, "timelapse_%06d.png")
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                input_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                output_video_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Timelapse video created successfully with ffmpeg: {output_video_path}")
                return True
            print(f"ffmpeg failed, trying OpenCV: {result.stderr}")
            raise Exception("ffmpeg failed")
        except Exception as ffmpeg_error:
            print(f"ffmpeg not available or failed, using OpenCV: {ffmpeg_error}")
            if len(frame_files) == 0:
                print("No frames to process")
                return False
            first_frame = cv2.imread(frame_files[0])
            if first_frame is None:
                print(f"Could not read first frame: {frame_files[0]}")
                return False
            height, width, _layers = first_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not video_writer.isOpened():
                print("Could not open video writer")
                return False
            for frame_file in tqdm(frame_files, desc="Creating video with OpenCV"):
                frame = cv2.imread(frame_file)
                if frame is not None:
                    video_writer.write(frame)
            video_writer.release()
            print(f"✓ Timelapse video created successfully with OpenCV: {output_video_path}")
            return True
    except Exception as e:
        print(f"✗ Error creating timelapse video: {e}")
        return False
