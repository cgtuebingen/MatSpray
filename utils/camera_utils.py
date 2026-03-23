import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode
from scene.cameras import Camera
from utils.graphics_utils import focal2fov

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_h, orig_w = cam_info.image.shape[:2]

    if args.resolution in [1, 2, 4, 8]:
        scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = global_down * resolution_scale
    resolution = (int(orig_h / scale), int(orig_w / scale))

    if cam_info.image is not None:
        image = torch.from_numpy(cam_info.image).float().permute(2, 0, 1)
        if scale == 1:
            resized_image_rgb = image
        else:
            resized_image_rgb = torchvision.transforms.Resize(resolution, antialias=True)(image)
        gt_image = resized_image_rgb

    resized_depth = None
    if cam_info.depth is not None:
        depth = torch.from_numpy(cam_info.depth).float().unsqueeze(0)
        resized_depth = torchvision.transforms.Resize(
            resolution, interpolation=InterpolationMode.NEAREST)(depth)

    resized_normal = None
    if cam_info.normal is not None:
        normal = torch.from_numpy(cam_info.normal).float().permute(2, 0, 1)
        resized_normal = torchvision.transforms.Resize(
            resolution, interpolation=InterpolationMode.NEAREST)(normal)

    resized_image_mask = None
    if cam_info.image_mask is not None:
        image_mask = torch.from_numpy(cam_info.image_mask).float().unsqueeze(0)
        resized_image_mask = torchvision.transforms.Resize(
            resolution, interpolation=InterpolationMode.NEAREST)(image_mask)

    # change the fx and fy
    scale_cx = cam_info.cx
    scale_cy = cam_info.cy
    scale_fx = cam_info.fx
    scale_fy = cam_info.fy
    if cam_info.cx is not None and cam_info.cy is not None:
        scale_cx /= scale
        scale_cy /= scale
        scale_fx /= scale
        scale_fy /= scale

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, fx=scale_fx, fy=scale_fy, cx=scale_cx, cy=scale_cy,
                  image=gt_image, depth=resized_depth, normal=resized_normal, image_mask=resized_image_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, image_path=cam_info.image_path)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos, desc="resolution scale: {}".format(resolution_scale), leave=False)):
        # for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]

    # Camera class uses image_width/image_height, not width/height
    width = getattr(camera, 'image_width', getattr(camera, 'width', None))
    height = getattr(camera, 'image_height', getattr(camera, 'height', None))
    
    if camera.cx is None:
        if camera.fx is not None:
            # fx/fy exist but cx/cy don't - write fx/fy to JSON
            camera_entry = {
                'id': id,
                'img_name': camera.image_name,
                'width': width,
                'height': height,
                'position': pos.tolist(),
                'rotation': serializable_array_2d,
                'fx': camera.fx,
                'fy': camera.fy,
            }
        else:
            # Only FoV available
            camera_entry = {
                'id': id,
                'img_name': camera.image_name,
                'width': width,
                'height': height,
                'position': pos.tolist(),
                'rotation': serializable_array_2d,
                'FoVx': camera.FovX,
                'FoVy': camera.FovY,
            }
    else:
        # Full intrinsics available (fx, fy, cx, cy)
        camera_entry = {
            'id': id,
            'img_name': camera.image_name,
            'width': width,
            'height': height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fx': camera.fx,
            'fy': camera.fy,
            'cx': camera.cx,
            'cy': camera.cy,
        }
    return camera_entry


def JSON_to_camera(json_cam):
    # Debug: print what's in the JSON (only first time)
    if not hasattr(JSON_to_camera, '_debug_logged'):
        print(f"[DEBUG JSON_to_camera] JSON keys: {list(json_cam.keys())}")
        print(f"[DEBUG JSON_to_camera] Has 'fx': {'fx' in json_cam}, value: {json_cam.get('fx', 'N/A')}")
        print(f"[DEBUG JSON_to_camera] Has 'fy': {'fy' in json_cam}, value: {json_cam.get('fy', 'N/A')}")
        print(f"[DEBUG JSON_to_camera] Has 'cx': {'cx' in json_cam}, value: {json_cam.get('cx', 'N/A')}")
        print(f"[DEBUG JSON_to_camera] Has 'cy': {'cy' in json_cam}, value: {json_cam.get('cy', 'N/A')}")
        print(f"[DEBUG JSON_to_camera] Has 'FoVx': {'FoVx' in json_cam}, value: {json_cam.get('FoVx', 'N/A')}")
        print(f"[DEBUG JSON_to_camera] Has 'FoVy': {'FoVy' in json_cam}, value: {json_cam.get('FoVy', 'N/A')}")
        JSON_to_camera._debug_logged = True
    
    rot = np.array(json_cam['rotation'])
    pos = np.array(json_cam['position'])
    W2C = np.zeros((4, 4))
    W2C[:3, :3] = rot
    W2C[:3, 3] = pos
    W2C[3, 3] = 1
    Rt = np.linalg.inv(W2C)
    R = Rt[:3, :3].transpose()
    T = Rt[:3, 3]
    H, W = json_cam['height'], json_cam['width']
    if 'cx' not in json_cam:
        if 'fx' in json_cam:
            # fx/fy exist but cx/cy don't - use default principal point (center of image)
            FovX = focal2fov(json_cam["fx"], W)
            FovY = focal2fov(json_cam["fy"], H)
            # Use image center as default principal point
            default_cx = W / 2.0
            default_cy = H / 2.0
            camera = Camera(colmap_id=0, R=R, T=T, FoVx=FovX, FoVy=FovY, 
                           fx=json_cam["fx"], fy=json_cam["fy"], cx=default_cx, cy=default_cy,
                           image=None, image_name=json_cam['img_name'], uid=json_cam['id'],
                           data_device='cuda', height=H, width=W)
            if not hasattr(JSON_to_camera, '_created_logged'):
                print(f"[DEBUG JSON_to_camera] Created camera with fx={json_cam['fx']}, fy={json_cam['fy']}, cx={default_cx}, cy={default_cy}")
        else:
            # Only FoV available, no intrinsics
            FovX = json_cam["FoVx"]
            FovY = json_cam["FoVy"]
            camera = Camera(colmap_id=0, R=R, T=T, FoVx=FovX, FoVy=FovY, fx=None, fy=None, cx=None, cy=None,
                           image=None, image_name=json_cam['img_name'], uid=json_cam['id'],
                           data_device='cuda', height=H, width=W)
            if not hasattr(JSON_to_camera, '_created_logged'):
                print(f"[DEBUG JSON_to_camera] Created camera with FoV only (fx=None, fy=None)")
        JSON_to_camera._created_logged = True
    else:
        # Full intrinsics available (fx, fy, cx, cy)
        FovX = focal2fov(json_cam["fx"], W)
        FovY = focal2fov(json_cam["fy"], H)
        camera = Camera(colmap_id=0, R=R, T=T, FoVx=FovX, FoVy=FovY, fx=json_cam["fx"], fy=json_cam["fy"],
                       cx=json_cam["cx"], cy=json_cam["cy"], image=None, image_name=json_cam['img_name'],
                       uid=json_cam['id'], data_device='cuda', height=H, width=W)
        if not hasattr(JSON_to_camera, '_created_logged'):
            print(f"[DEBUG JSON_to_camera] Created camera with full intrinsics: fx={json_cam['fx']}, fy={json_cam['fy']}, cx={json_cam['cx']}, cy={json_cam['cy']}")
        JSON_to_camera._created_logged = True
    
    # Verify what was actually set on the camera object
    if not hasattr(JSON_to_camera, '_verify_logged'):
        print(f"[DEBUG JSON_to_camera] Camera object after creation:")
        print(f"  camera.fx = {camera.fx}")
        print(f"  camera.fy = {camera.fy}")
        print(f"  camera.cx = {camera.cx}")
        print(f"  camera.cy = {camera.cy}")
        print(f"  camera.FoVx = {camera.FoVx}")
        print(f"  camera.FoVy = {camera.FoVy}")
        JSON_to_camera._verify_logged = True
    
    return camera
