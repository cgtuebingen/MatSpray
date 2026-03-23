base_path="${NAVI_RAW_DIR:-/path/to/navi/v1.2/v1.2/}"
dataset_path="${NAVI_DATASET_DIR:-/path/to/datasets/navi/}"

name="well_with_leaf_roof_showpiece"
camera_version="00_pixel_4a"

rm -rf $dataset_path/$name/base

python3 "$(dirname "$0")/../utils/downsample_and_apply_masks.py" \
  --mode both \
  --images_dir $base_path/$name/multiview_$camera_version/images/ \
  --masks_dir $base_path/$name/multiview_$camera_version/mask/ \
  --scale 8 \
  --output_images_dir $dataset_path/$name/base/images \
  --output_unmasked_dir $dataset_path/$name/base/images_bg \
  --output_masks_dir $dataset_path/$name/base/masks   # optional

# COLMAP manual pipeline (features → matches → mapping)
workspace="$dataset_path/$name/base"
image_dir="$workspace/images_bg"   # switch to "$workspace/images" if you prefer masked images

mkdir -p "$workspace/sparse"
rm -f "$workspace/database.db"

# Feature extraction (more features + DSP)
colmap feature_extractor \
  --database_path "$workspace/database.db" \
  --image_path "$image_dir" \
  --ImageReader.camera_model SIMPLE_PINHOLE \
  --ImageReader.single_camera=1 \
  --SiftExtraction.max_num_features=16000 \
  --SiftExtraction.domain_size_pooling=1

# Exhaustive matching with guided matching; optionally relax ratio test a bit
colmap exhaustive_matcher \
  --database_path "$workspace/database.db" \
  --SiftMatching.guided_matching=1 \
  --SiftMatching.max_ratio=0.9

# Mapping: slightly lower init inlier threshold
colmap mapper \
  --database_path "$workspace/database.db" \
  --image_path "$image_dir" \
  --output_path "$workspace/sparse" \
  --Mapper.init_min_num_inliers=20
