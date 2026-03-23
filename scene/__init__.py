#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, JSON_to_camera


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        # --- PATCH: Directly load cameras.json using JSON_to_camera if present ---
        if os.path.exists(os.path.join(args.source_path, "cameras.json")):
            print("Found cameras.json file, loading Gaussian Splatting cameras (direct JSON_to_camera)!")
            cameras_json_path = os.path.join(args.source_path, "cameras.json")
            with open(cameras_json_path, 'r') as f:
                cameras_data = json.load(f)
            camera_objs = [JSON_to_camera(cam_data) for cam_data in cameras_data]
            # Try to attach image paths so lazy loading can find ground-truth
            for cam_data, cam_obj in zip(cameras_data, camera_objs):
                img_stem = cam_data.get('img_name')
                candidate_dirs = [
                    os.path.join(args.source_path, 'train', 'images'),
                    os.path.join(args.source_path, 'train', 'images_bg'),
                    os.path.join(args.source_path, 'images'),
                    os.path.join(args.source_path, 'images_bg'),
                ]
                candidate_exts = ['.png', '.jpg', '.jpeg', '.exr']
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
                    # Fallback: scan for any file starting with the stem
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
                setattr(cam_obj, 'image_path', found_path)
            self.train_cameras[1.0] = camera_objs
            self.test_cameras[1.0] = []  # Or split if you want
            # Compute extent from cameras if needed, else set to 1.0
            self.cameras_extent = 1.0
            self.scene_info = None
            return  # Skip the rest of the constructor
        # --- END PATCH ---

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval,
                                                          debug=args.debug_cuda)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            # Read the entire file to check for OpenCV format
            with open(os.path.join(args.source_path, "transforms_train.json"), 'r') as f:
                content = f.read()
                print(f"DEBUG: Checking transforms file for camera model...")
                print(f"DEBUG: First 200 chars: {content[:200]}")
                if '"camera_model": "OPENCV"' in content:
                    print("Found transforms_train.json file with OpenCV camera model!")
                    scene_info = sceneLoadTypeCallbacks["OpenCV"](args.source_path, args.white_background, args.eval, 
                                                                   debug=args.debug_cuda)
                elif "Synthetic4Relight" in args.source_path:
                    print("Found transforms_train.json file, assuming Synthetic4Relight data set!")
                    scene_info = sceneLoadTypeCallbacks["Synthetic4Relight"](args.source_path, args.white_background, args.eval,
                                                                debug=args.debug_cuda)
                else:
                    print("Found transforms_train.json file, assuming Blender data set!")
                    scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, 
                                                                   debug=args.debug_cuda)
        
        elif os.path.exists(os.path.join(args.source_path, "inputs/sfm_scene.json")):
            print("Found sfm_scene.json file, assuming NeILF data set!")
            scene_info = sceneLoadTypeCallbacks["NeILF"](args.source_path, args.white_background, args.eval,
                                                         debug=args.debug_cuda)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        self.scene_info = scene_info

    def save(self, iteration, final_iteration=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), 
                               current_iteration=iteration, 
                               final_iteration=final_iteration)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
