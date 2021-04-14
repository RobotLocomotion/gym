import os
from pydrake.common import FindResourceOrThrow

pusher_peg = os.path.join(os.path.dirname(__file__), "assets/pusher_peg.urdf")
model_paths = dict({
    'cracker_box': FindResourceOrThrow(
        "drake/manipulation/models/ycb/sdf/003_cracker_box.sdf"),
    'sugar_box': FindResourceOrThrow(
        "drake/manipulation/models/ycb/sdf/004_sugar_box.sdf"),
    'tomato_soup_can': FindResourceOrThrow(
        "drake/manipulation/models/ycb/sdf/005_tomato_soup_can.sdf"),
    'mustard_bottle': FindResourceOrThrow(
        "drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf"),
    'gelatin_box': FindResourceOrThrow(
        "drake/manipulation/models/ycb/sdf/009_gelatin_box.sdf"),
    'potted_meat_can': FindResourceOrThrow(
        "drake/manipulation/models/ycb/sdf/010_potted_meat_can.sdf"),
    'plate': os.path.join(os.path.dirname(__file__),
                          "assets/plate/plate_11in_decomp.urdf"),
})
model_baselink_names = dict({
    'cracker_box': 'base_link_cracker',
    'sugar_box': 'base_link_sugar',
    'tomato_soup_can': 'base_link_soup',
    'mustard_bottle': 'base_link_mustard',
    'gelatin_box': 'base_link_gelatin',
    'potted_meat_can': 'base_link_meat',
    'plate': 'plate_11in_decomp_body_link',
})