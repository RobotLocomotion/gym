import os
import numpy as np
import yaml
from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage

from pydrake.geometry import Box
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import CoulombFriction
from pydrake.multibody.tree import UnitInertia, SpatialInertia
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import PidController
from pydrake.systems.framework import DiagramBuilder

import gym
from gym import error, spaces, utils
from gym.envs.robot_locomotion_group.drake.drake_sim_diagram import \
    DrakeSimDiagram
from gym.envs.robot_locomotion_group.drake.pusher_slider import path_util


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(dict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        space = spaces.Box(-float('inf'), float('inf'), shape=observation.shape,
                           dtype=observation.dtype)
    elif isinstance(observation, float):
        space = spaces.Box(-float('inf'), float('inf'), shape=(1,), dtype=float)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class PusherSliderEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, config=None):
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            config = yaml.safe_load(open(config_path, 'r'))

        self._config = config
        self._step_dt = config["step_dt"]
        self._model_name = config["model_name"]

        self._sim_diagram = DrakeSimDiagram(config)
        mbp = self._sim_diagram.mbp
        builder = self._sim_diagram.builder
        # === Add table =====
        dims = config["table"]["size"]
        color = np.array(config["table"]["color"])
        friction_params = config["table"]["coulomb_friction"]
        box_shape = Box(*dims)
        X_W_T = RigidTransform(p=np.array([0., 0., -dims[2]/2.]))

        # This rigid body will be added to the world model instance since
        # the model instance is not specified.
        box_body = mbp.AddRigidBody("table", SpatialInertia(
            mass=1.0, p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
        mbp.WeldFrames(mbp.world_frame(), box_body.body_frame(), X_W_T)
        mbp.RegisterVisualGeometry(
            box_body, RigidTransform(), box_shape, "table_vis", color)
        mbp.RegisterCollisionGeometry(
            box_body, RigidTransform(), box_shape, "table_collision",
            CoulombFriction(*friction_params))

        # === Add pusher ====
        parser = Parser(mbp, self._sim_diagram.sg)
        self._pusher_model_id = parser.AddModelFromFile(path_util.pusher_peg, "pusher")
        base_link = mbp.GetBodyByName("base", self._pusher_model_id)
        mbp.WeldFrames(mbp.world_frame(), base_link.body_frame())

        def pusher_port_func():
            actuation_input_port = mbp.get_actuation_input_port(self._pusher_model_id)
            state_output_port = mbp.get_state_output_port(self._pusher_model_id)
            builder.ExportInput(actuation_input_port, "pusher_actuation_input")
            builder.ExportOutput(state_output_port, "pusher_state_output")
        self._sim_diagram.finalize_functions.append(pusher_port_func)

        # === Add slider ====
        parser = Parser(mbp, self._sim_diagram.sg)
        self._slider_model_id = parser.AddModelFromFile(
            path_util.model_paths[self._model_name], self._model_name)

        def slider_port_func():
            state_output_port = mbp.get_state_output_port(self._slider_model_id)
            builder.ExportOutput(state_output_port, "slider_state_output")
        self._sim_diagram.finalize_functions.append(slider_port_func)

        if "rgbd_sensors" in config and config["rgbd_sensors"]["enabled"]:
            self._sim_diagram.add_rgbd_sensors_from_config(config)

        if "visualization" in config:
            if not config["visualization"]:
                pass
            elif config["visualization"] == "meshcat":
                self._sim_diagram.connect_to_meshcat()
            elif config["visualization"] == "drake_viz":
                self._sim_diagram.connect_to_drake_visualizer()
            else:
                raise ValueError("Unknown visualization:", config["visualization"])

        self._sim_diagram.finalize()

        builder = DiagramBuilder()
        builder.AddSystem(self._sim_diagram)

        pid = builder.AddSystem(PidController(kp=[0, 0], kd=[1000, 1000], ki=[0, 0]))
        builder.Connect(self._sim_diagram.GetOutputPort("pusher_state_output"),
                        pid.get_input_port_estimated_state())
        builder.Connect(pid.get_output_port_control(),
                        self._sim_diagram.GetInputPort("pusher_actuation_input"))
        builder.ExportInput(pid.get_input_port_desired_state(),
                            "pid_input_port_desired_state")

        self._diagram = builder.Build()
        self._pid_desired_state_port = self._diagram.get_input_port(0)
        self._simulator = Simulator(self._diagram)
        self.reset()

        self.action_space = spaces.Box(low=-1., high= 1., shape=(2,),
                                       dtype=np.float32)
        # TODO: Observation space for images is currently too loose
        self.observation_space = convert_observation_to_space(self.get_observation())

    def get_observation(self, context=None):
        assert self._sim_diagram.is_finalized()
        if context is None:
            context = self._simulator.get_context()
        context = self._diagram.GetMutableSubsystemContext(self._sim_diagram, context)
        mbp_context = self._sim_diagram.GetMutableSubsystemContext(
            self._sim_diagram.mbp, context)
        obs = dict()
        obs["time"] = context.get_time()
        obs["pusher"] = dict()
        obs["pusher"]["position"] = np.copy(self._sim_diagram.mbp.GetPositions(
            mbp_context, self._pusher_model_id))
        obs["pusher"]["velocity"] = np.copy(self._sim_diagram.mbp.GetVelocities(
            mbp_context, self._pusher_model_id))
        q = np.copy(self._sim_diagram.mbp.GetPositions(mbp_context,
                                                       self._slider_model_id))
        v = np.copy(self._sim_diagram.mbp.GetVelocities(mbp_context,
                                                        self._slider_model_id))
        obs["slider"] = dict()
        obs["slider"]["position"] = {'translation': q[4:], 'quaternion': q[0:4], 'raw': q}
        obs["slider"]["velocity"] = {'linear': v[3:], 'angular': v[0:3], 'raw': v}
        if self._config["rgbd_sensors"]["enabled"]:
            obs["images"] = self._sim_diagram.get_image_observations(context)
        return obs

    def pusher_slider_on_table(self, context=None):
        if context is None:
            context = self._simulator.get_context()
        context = self._diagram.GetMutableSubsystemContext(self._sim_diagram, context)
        mbp_context = self._sim_diagram.GetMutableSubsystemContext(
            self._sim_diagram.mbp, context)

        table_dim = np.array(self._config["table"]["size"])
        q_pusher = self._sim_diagram.mbp.GetPositions(mbp_context,
                                                      self._pusher_model_id)
        q_slider = self._sim_diagram.mbp.GetPositions(
            mbp_context, self._slider_model_id)[4:6]

        tol = 0.03
        high_edge = table_dim[:2]/2.0 - tol
        low_edge = -high_edge

        return ((low_edge < q_pusher) & (q_pusher < high_edge) &
                (low_edge < q_slider) & (q_slider < high_edge)).all()

    def step(self, action, dt=None):
        assert self._sim_diagram.is_finalized()
        assert len(action) == 2
        if dt is None:
            dt = self._step_dt

        # the time to advance to
        t_advance = self._simulator.get_context().get_time() + dt
        context = self._simulator.get_mutable_context()

        # set the value of the PID controller input port
        pid_setpoint = np.concatenate((np.zeros(2), action))
        self._pid_desired_state_port.FixValue(context, pid_setpoint)

        # simulate and take observation
        self._simulator.AdvanceTo(t_advance)
        obs = self.get_observation()

        reward = 0.
        done = self.pusher_slider_on_table()
        info = {}

        return obs, reward, done, info

    # TODO: Make reset position non-deterministic?
    def reset(self, q_pusher=None, q_slider=None):
        assert self._sim_diagram.is_finalized()
        if q_pusher is None:
            q_pusher = np.array([-0.1, 0.])
        if q_slider is None:
            q_slider = np.array([1., 0., 0., 0., 0., 0., 0.05])
        pid_setpoint = np.zeros(4)

        context = self._simulator.get_mutable_context()
        mbp_context = self._diagram.GetMutableSubsystemContext(
            self._sim_diagram.mbp, context)
        context.SetTime(0.)
        self._sim_diagram.mbp.SetPositions(mbp_context, self._pusher_model_id,
                                           q_pusher)
        self._sim_diagram.mbp.SetVelocities(mbp_context, self._pusher_model_id,
                                            np.zeros(2))
        self._sim_diagram.mbp.SetPositions(mbp_context, self._slider_model_id,
                                           q_slider)
        self._sim_diagram.mbp.SetVelocities(mbp_context, self._slider_model_id,
                                            np.zeros(6))
        self._pid_desired_state_port.FixValue(context, pid_setpoint)
        self._simulator.Initialize()
        return self.get_observation()

    #TODO: Implement as necessary
    def render(self, mode="human"):
        pass

    def close(self):
        pass

    #TODO: Implement?
    def seed(self, seed=None):
        pass

    def get_labels_to_mask(self):
        """
        Returns a set of labels that represent slider object in the scene
        :return: TinyDB database
        """
        # write sql query
        label_db = self._sim_diagram.get_label_db()
        # return all objects that have matching model_instance_id in the list
        Label = Query()
        labels_to_mask = label_db.search(Label.model_instance_id.one_of(
            int(self._slider_model_id)))
        # create new TinyDB with just those
        mask_db = TinyDB(storage=MemoryStorage)
        mask_db.insert_multiple(labels_to_mask)

        return mask_db
