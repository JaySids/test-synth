import blenderproc as bproc
import numpy as np
import random
import argparse
import os
import logging

# Set up verbose logging
logging.basicConfig(level=logging.DEBUG)

_VALID_DEGREES = [d for d in range(360) if d % 90 != 0]
def gen_rotation():
    degree = random.choice(_VALID_DEGREES)
    return np.deg2rad(degree)

parser = argparse.ArgumentParser()
parser.add_argument('scene', nargs='?', default="test/sambar.obj")
parser.add_argument('--runs', '-r', type=int, default=10)
args = parser.parse_args()

# Global init
bproc.init()
bproc.camera.set_resolution(1920, 1080)
bproc.python.camera.CameraUtility.set_intrinsics_from_blender_params(
    lens=28.22, lens_unit="MILLIMETERS"
)

# Load scene
objs = bproc.loader.load_obj(args.scene)

# Set static camera pose
cam_pose = bproc.math.build_transformation_mat([0,0,96.773], [0,0,0])
bproc.camera.add_camera_pose(cam_pose)

# Filter objects
lid = bproc.filter.one_by_attr(objs, "name", "Lid")
background = bproc.filter.one_by_attr(objs, "name", "backframe")

# Set object properties
background.set_location([0, 0, 0])
background.set_rotation_euler([np.deg2rad(90), 0, 0])
background.set_scale([1, 1, 1])

# ⬇️ PRIMARY FIX: Use a unique ID for the lid
lid.set_cp("category_id", 1) 
lid.set_cp("supercategory", "object")
lid.set_scale([0.969196, 0.969196, 0.969196])

for run_id in range(args.runs):
    # Set up light for this run
    pt_light = bproc.types.Light()
    pt_light.set_type("SUN")
    pt_light.set_energy(0.01)
    pt_light.set_color([182, 182, 171])
    pt_light.set_location([3.03904, 27.7728, 79.871])
    pt_light.set_scale([-11.0794, -1.84018, -11.0794])
    pt_light.set_rotation_euler([np.deg2rad(-232.469), np.deg2rad(15.476), np.deg2rad(1.7676)])
    lid.set_location([0, 0, 0])

    # ⬇️ SECONDARY FIX: Adjust Z-location to prevent Z-fighting
    lid.set_location(np.random.uniform(
        [-54.5335 * 0.8, -22.7494 * 0.8, 0.1],
        [51.411 * 0.8, 26.5257  * 0.8, 0.1]
    ).tolist())
    lid.set_rotation_euler([np.deg2rad(90), 0, gen_rotation()])

    # Enable segmentation with the correct default ID
    bproc.renderer.enable_segmentation_output(
        map_by=["category_id","instance","name"],
        default_values={"category_id": 0} # Background pixels will have ID 1
    )
    
    # Render and write data
    data = bproc.renderer.render()
    bproc.writer.write_coco_annotations(
        os.path.join("output","coco_data"),
        instance_segmaps=data["instance_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        colors=data["colors"],
        color_file_format="JPEG",
        mask_encoding_format="polygon"
    )

    # Clean up the light
    pt_light.delete()