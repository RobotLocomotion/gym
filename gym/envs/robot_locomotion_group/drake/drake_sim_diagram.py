from future.utils import iteritems
import numpy as np
from tinydb import TinyDB
from tinydb.storages import MemoryStorage

from pydrake.geometry import DrakeVisualizer
from pydrake.geometry.render import ClippingRange, ColorRenderCamera, \
    DepthRange, DepthRenderCamera, RenderCameraCore, RenderEngineVtkParams, \
    MakeRenderEngineVtk
from pydrake.math import RigidTransform
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import BodyIndex
from pydrake.systems.framework import DiagramBuilder, Diagram
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.systems.sensors import CameraInfo, RgbdSensor

from gym.envs.robot_locomotion_group.drake.transform_utils import \
    transform_from_dict


class DrakeSimDiagram(Diagram):
    def __init__(self, config):
        Diagram.__init__(self)
        
        dt = config["mbp_dt"]
        self._builder = DiagramBuilder()
        self._mbp, self._sg = AddMultibodyPlantSceneGraph(self._builder, dt)
        
        self._finalize_functions = []
        self._finalized = False
        self._rgbd_sensors = dict()
        self._renderer_name = None
        
    # === Property accessors ========================================
    @property
    def mbp(self):
        return self._mbp

    @property
    def sg(self):
        return self._sg

    @property
    def builder(self):
        return self._builder

    @property
    def finalize_functions(self):
        return self._finalize_functions

    @property
    def rgbd_sensors(self):
        return self._rgbd_sensors

    # === Add visualizers ===========================================
    def connect_to_meshcat(self):
        self._meshcat = ConnectMeshcatVisualizer(
            self._builder, scene_graph=self._sg,
            zmq_url="tcp://127.0.0.1:6000", draw_period=1)
        return self._meshcat

    def connect_to_drake_visualizer(self):
        self._drake_viz = DrakeVisualizer.AddToBuilder(
            builder=self._builder, scene_graph=self._sg)
        return self._drake_viz
    
    # === Add Cameras ===============================================
    def add_rgbd_sensors_from_config(self, config):
        if not config["rgbd_sensors"]["enabled"]:
            return
        for camera_name, sensor_config in iteritems(config["rgbd_sensors"]["sensor_list"]):
            self.add_rgbd_sensor(camera_name, sensor_config)

    def add_rgbd_sensor(self, camera_name, sensor_config):
        """
        Adds Rgbd camera to the diagram
        """
        builder = self._builder
        if self._renderer_name is None:
            self._renderer_name = "vtk_renderer"
            self._sg.AddRenderer(self._renderer_name,
                                 MakeRenderEngineVtk(RenderEngineVtkParams()))

        width = sensor_config['width']
        height = sensor_config['height']
        fov_y = sensor_config['fov_y']
        z_near = sensor_config['z_near']
        z_far = sensor_config['z_far']

        # This is in right-down-forward convention
        X_W_camera = transform_from_dict(sensor_config)
        color_camera = ColorRenderCamera(RenderCameraCore(
                self._renderer_name,
                CameraInfo(width, height, fov_y),
                ClippingRange(z_near, z_far),
                RigidTransform()
            ), False)
        depth_camera = DepthRenderCamera(color_camera.core(), DepthRange(z_near, z_far))

        # add camera system
        camera = builder.AddSystem(RgbdSensor(
            parent_id=self._sg.world_frame_id(), X_PB=X_W_camera,
            color_camera=color_camera, depth_camera=depth_camera))
        builder.Connect(self._sg.get_query_output_port(),
                        camera.query_object_input_port())

        self._rgbd_sensors[camera_name] = camera

    # === Finalize the completed diagram ============================
    def finalize(self):
        self._mbp.Finalize()
        self._finalized = True

        for func in self._finalize_functions:
            func()

        self._builder.ExportOutput(self._sg.get_pose_bundle_output_port(),
                                   "pose_bundle")
        self._builder.ExportOutput(self._mbp.get_contact_results_output_port(),
                                   "contact_results")
        self._builder.ExportOutput(self._mbp.get_state_output_port(),
                                   "plant_continuous_state")
        self._builder.ExportOutput(self._mbp.get_geometry_poses_output_port(),
                                   "geometry_poses")
        self._builder.ExportInput(self._mbp.get_applied_spatial_force_input_port(),
                                  "spatial_input")
        self._builder.ExportOutput(self._mbp.get_body_poses_output_port(),
                                   "body_poses")

        self._builder.BuildInto(self)

    def is_finalized(self):
        return self._finalized

    # === Camera helpers ============================================

    def get_image_observations_single_sensor(self, sensor_name, context):
        assert self.is_finalized()
        sensor = self._rgbd_sensors[sensor_name]
        sensor_context = self.GetSubsystemContext(sensor, context)
        rgb = sensor.color_image_output_port().Eval(sensor_context)
        depth_32F = sensor.depth_image_32F_output_port().Eval(sensor_context)
        depth_16U = sensor.depth_image_16U_output_port().Eval(sensor_context)
        label = sensor.label_image_output_port().Eval(sensor_context)

        return {'rgb': np.copy(rgb.data[:, :, :3]),
                'depth_32F': np.copy(depth_32F.data).squeeze(),
                'depth_16U': np.copy(depth_16U.data).squeeze(),
                'label': np.copy(label.data).squeeze()}

    def get_image_observations(self, context):
        image_dict = dict()
        for sensor_name in self._rgbd_sensors:
            image_dict[sensor_name] = self.get_image_observations_single_sensor(
                sensor_name, context)
        return image_dict

    def get_label_db(self):
        """
        Builds database that associates bodies and labels
        :return: TinyDB database
        """
        db = TinyDB(storage=MemoryStorage)
        for i in range(self._mbp.num_bodies()):
            body = self._mbp.get_body(BodyIndex(i))
            model_instance_id = int(body.model_instance())

            body_name = body.name()
            model_name = self._mbp.GetModelInstanceName(body.model_instance())

            entry = {'label': i,
                     'model_instance_id': model_instance_id,
                     'body_name': body_name,
                     'model_name': model_name}
            db.insert(entry)

        return db
