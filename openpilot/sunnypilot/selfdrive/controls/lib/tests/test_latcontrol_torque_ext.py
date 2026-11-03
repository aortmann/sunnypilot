"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import numpy as np

from openpilot.cereal import log, messaging
from opendbc.car.structs import car
from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.values import CAR as HONDA
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.locationd.helpers import Pose
from openpilot.common.mock.generators import generate_deviceMotion
from openpilot.sunnypilot.selfdrive.car import interfaces as sunnypilot_interfaces
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.common.test import OpenpilotTestCase


def _make_controller(enhanced=False, nnlc=False):
  params = Params()
  params.put_bool("EnforceTorqueControl", True, block=True)
  params.put_bool("LateralJerkTorqueController", enhanced, block=True)
  params.put_bool("NeuralNetworkLateralControl", nnlc, block=True)

  car_name = HONDA.HONDA_CIVIC
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
  CI = CarInterface(CP, CP_SP)
  sunnypilot_interfaces.setup_interfaces(CI, params)
  CP_SP = convert_to_capnp(CP_SP)
  VM = VehicleModel(CP)
  controller = LatControlTorque(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)
  return controller, VM, CP


def _make_model_v2():
  model = messaging.new_message('modelV2')
  position = log.XYZTData.new_message()
  position.x = [float(x) for x in 30.0 * np.array(ModelConstants.T_IDXS)]
  model.modelV2.position = position
  orientation = log.XYZTData.new_message()
  orientation.x = [0.0 for _ in ModelConstants.T_IDXS]
  orientation.y = [0.0 for _ in ModelConstants.T_IDXS]
  model.modelV2.orientation = orientation
  velocity = log.XYZTData.new_message()
  velocity.x = [30.0 for _ in ModelConstants.T_IDXS]
  model.modelV2.velocity = velocity
  acceleration = log.XYZTData.new_message()
  acceleration.x = [0.0 for _ in ModelConstants.T_IDXS]
  acceleration.y = [0.0 for _ in ModelConstants.T_IDXS]
  model.modelV2.acceleration = acceleration
  return model


def _run_update(controller, VM):
  CS = car.CarState.new_message()
  CS.vEgo = 30
  CS.steeringPressed = False
  lp = generate_deviceMotion()
  pose = Pose.from_device_motion(lp.deviceMotion)
  params = log.VehicleParameters.new_message()
  model_v2 = _make_model_v2().modelV2
  controller.extension.update_model_v2(model_v2)
  controller.extension.update_lateral_lag(0.2)
  return controller.update(True, CS, VM, params, False, 0.5, pose, False, 0.2)


class TestLatControlTorqueExt(OpenpilotTestCase):
  def test_init_enhanced_only(self):
    controller, VM, _ = _make_controller(enhanced=True, nnlc=False)
    assert controller.extension._jerk_aware_enabled
    assert not controller.extension.enabled  # NNLC disabled

  def test_init_nnlc_only(self):
    controller, VM, _ = _make_controller(enhanced=False, nnlc=True)
    assert not controller.extension._jerk_aware_enabled
    assert controller.extension.enabled

  def test_init_neither(self):
    controller, VM, _ = _make_controller(enhanced=False, nnlc=False)
    assert not controller.extension._jerk_aware_enabled
    assert not controller.extension.enabled

  def test_init_both_no_crash(self):
    controller, VM, _ = _make_controller(enhanced=True, nnlc=True)
    assert not controller.extension._jerk_aware_enabled
    assert not controller.extension.enabled

  def test_update_enhanced_only(self):
    controller, VM, _ = _make_controller(enhanced=True, nnlc=False)
    output_torque, _, pid_log = _run_update(controller, VM)
    assert pid_log.active

  def test_update_neither(self):
    controller, VM, _ = _make_controller(enhanced=False, nnlc=False)
    output_torque, _, pid_log = _run_update(controller, VM)
    assert pid_log.active

  def test_update_both_no_crash(self):
    controller, VM, _ = _make_controller(enhanced=True, nnlc=True)
    output_torque, _, pid_log = _run_update(controller, VM)
    assert pid_log.active

  def test_nnlc_pid_is_independent_of_host(self):
    # the NNLC extension runs its own constant-gain PID in torque space, not the host's speed-scheduled one
    controller, VM, _ = _make_controller(enhanced=False, nnlc=True)
    ext = controller.extension
    assert ext._pid is not controller.pid
    assert ext._pid.pos_limit == 1.0 and ext._pid.neg_limit == -1.0
    for speed in (1.0, 2.0, 5.0, 30.0):
      ext._pid.speed = speed
      assert ext._pid.k_p == 1.0
      assert ext._pid.k_i == 0.3

  def test_jerk_aware_pid_keeps_host_gain_schedule(self):
    controller, VM, _ = _make_controller(enhanced=True, nnlc=False)
    ext = controller.extension
    assert ext._pid is not controller.pid
    assert ext._pid.pos_limit == 1.0 and ext._pid.neg_limit == -1.0
    for speed in (1.0, 2.0, 5.0, 30.0):
      ext._pid.speed = controller.pid.speed = speed
      assert ext._pid.k_p == controller.pid.k_p
      assert ext._pid.k_i == controller.pid.k_i

  def test_update_logs_extension_pid(self):
    controller, VM, _ = _make_controller(enhanced=True, nnlc=False)
    output_torque, _, pid_log = _run_update(controller, VM)
    ext = controller.extension
    # the log fields are float32
    assert math.isclose(pid_log.p, ext._pid.p, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(pid_log.i, ext._pid.i, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(pid_log.f, ext._pid.f, rel_tol=1e-6, abs_tol=1e-6)
    assert output_torque == -float(ext._pid.control)

  def test_update_nnlc_logs_extension_pid(self):
    controller, VM, _ = _make_controller(enhanced=False, nnlc=True)
    output_torque, _, pid_log = _run_update(controller, VM)
    ext = controller.extension
    assert ext._nnlc_enabled
    assert pid_log.p == float(ext._pid.p)
    assert pid_log.p == pid_log.error  # KP = 1 in torque space
    assert output_torque == -float(ext._pid.control)

  def test_reset_clears_extension_integrator(self):
    controller, VM, _ = _make_controller(enhanced=True, nnlc=False)
    controller.extension._pid.i = 0.1
    controller.reset()
    assert controller.extension._pid.i == 0.0
