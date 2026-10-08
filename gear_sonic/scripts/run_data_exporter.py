"""
Sonic VLA data exporter for G1 -- NO ROS 2 DEPENDENCY.

All data sources use ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (port 5557, from C++ zmq_output_handler)
  2. SMPL pose    -> ZMQ SUB on ``pose`` topic     (port 5556, from pico_manager_thread_server)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor

Robot config (``script_config`` in info.json) is read from the ``robot_config``
ZMQ topic re-published every ~2 s by the C++ process.  If the config is not
received within the timeout the exporter exits with an error.

Virtual environment setup (run from repo root):
    bash install_scripts/install_data_collection.sh
    source .venv_data_collection/bin/activate

Usage (from repo root):
    python gear_sonic/scripts/run_data_exporter.py --task-prompt "pick up the cup"
    python gear_sonic/scripts/run_data_exporter.py --task-prompt "walk forward" --dataset-name my_session
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro
import zmq

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import (
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
    get_wrist_camera_features,
    get_wrist_camera_modality_config,
)
from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.camera.sensor_server import ImageMessageSchema
from gear_sonic.utils.data_collection.episode_state import EpisodeState
from gear_sonic.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.text_to_speech import TextToSpeech
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity, quat_to_rot6d
from gear_sonic.utils.data_collection.zmq_state_subscriber import (
    ZMQStateSubscriber,
    poll_robot_config_zmq,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SonicDataExporterConfig:
    """CLI config for the ROS-free Sonic data exporter."""

    # Dataset
    dataset_name: str | None = None
    """Dataset name (auto-generated if creating new)."""

    task_prompt: str = "demo"
    """Language task prompt."""

    root_output_dir: str = "outputs"
    """Root output directory."""

    data_collection_frequency: int = 50
    """Data collection frequency (Hz)."""


    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Sonic / SMPL pose (from pico_manager_thread_server)
    sonic_zmq_host: str = "localhost"
    """ZMQ host for Sonic SMPL pose messages."""

    sonic_zmq_port: int = 5556
    """ZMQ port for Sonic SMPL pose messages."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # Robot config
    robot_config_timeout: float = 0
    """Seconds to wait for the ZMQ robot_config message at startup (0 = wait forever)."""

    record_wrist_cameras: bool = False
    """Record wrist camera streams (left_wrist, right_wrist). Requires cameras to be available."""

    text_to_speech: bool = True
    """Use text-to-speech voice feedback."""

    # coScene dual recording (all off by default = upstream behaviour)
    keyboard_port: int = 5580
    """ZMQ port of the start/stop source. Besides the upstream c/x keys it takes idempotent
    'start <take id>' / 'stop <take id>' / 'discard <take id>' commands (sent by pico_record.py)."""

    ignore_manager_toggles: bool = False
    """Ignore the Manager's grip+A toggle_data_collection/toggle_data_abort flags, so the keyboard port is the
    only start/stop source; manager_state is then read on its own CONFLATE socket (latest stream_mode only), so
    the ~kHz manager_state stream cannot crowd pose/planner out of the 20-message queue."""

    eef_zmq_port: int = 0
    """If > 0: read the gripper from eef_zmq_tap.py on this port and add the observation.gripper /
    action.gripper columns (NaN when no sample younger than 0.5 s)."""

    frame_clock: bool = False
    """Rows on a fixed 1/fps grid of the robot's clock: the loop runs on absolute deadlines, a tick it missed
    becomes a held row (a repeat of the previous row, held=1) instead of a missing one, so row timestamp =
    frame_index/fps stays true to wall time. Adds the columns wall_time, held, state_time, camera_time
    (FRAME_CLOCK_FEATURES) and t0_wall to coscene_takes.jsonl. 'start/stop <take id> <unix time>' cut the
    episode at those times (pre-roll up to 1 s, rows emitted 0.1 s behind the newest tick)."""

    lean: bool = False
    """Less CPU, same output: decode only the camera images the dataset records, and only for rows it writes
    (nothing while idle; the robot's camera message carries three images, the dataset records one)."""

    video_preset: str = ""
    """libx264 preset of the episode video (e.g. 'veryfast'); empty = libx264's default (medium)."""

    video_threads: int = 0
    """libx264 threads; 0 = libx264's default (about 1.5 per core)."""


# Frame-clock columns (--frame-clock); UNIX seconds on the robot's wall clock
FRAME_CLOCK_FEATURES = {
    # when this row's inputs were read (a held row repeats the time of the read it repeats)
    "wall_time": {"dtype": "float64", "shape": (1,), "names": ["wall_time"]},
    # 1 = the loop missed this tick and the row repeats the previous row (its video frame too); 0 = fresh read
    "held": {"dtype": "int32", "shape": (1,), "names": ["held"]},
    # the controller's stamp of the g1_debug message in this row (its receive time when the controller sends 0)
    "state_time": {"dtype": "float64", "shape": (1,), "names": ["state_time"]},
    # the camera server's capture stamp of the first recorded image (ego_view); NaN if the message has none.
    # (No "image" in these names: lerobot's stats check treats every feature named *image* as a picture.)
    "camera_time": {"dtype": "float64", "shape": (1,), "names": ["camera_time"]},
}


class LazyImages:
    """The images of one camera message, each decoded on first use (--lean). The camera message carries more
    images than the dataset records (ego_view plus two mono views on the robot), and none is needed while idle."""

    def __init__(self, raw: dict):
        self._raw = raw
        self._decoded = {}

    def __contains__(self, key) -> bool:
        return key in self._raw

    def keys(self):
        return self._raw.keys()

    def __getitem__(self, key):
        if key not in self._decoded:
            self._decoded[key] = ImageMessageSchema.deserialize({"images": {key: self._raw[key]}}).images[key]
        return self._decoded[key]


@dataclass
class _Tick:
    """The latest inputs as read at one tick of the frame clock (references, never copies)."""

    k: int
    wall: float
    proprio: dict | None
    sonic: dict | None
    planner: dict | None
    stream_mode: int
    image: dict | None
    eef: dict | None


# Gripper columns from eef_zmq_tap.py (DM4310 jaw motor, rad; opening in mm; NaN = no fresh sample)
GRIPPER_FEATURES = {
    "observation.gripper": {
        "dtype": "float32",
        "shape": (4,),
        "names": ["left_q", "right_q", "left_opening_mm", "right_opening_mm"],
    },
    "action.gripper": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["left_cmd_q", "right_cmd_q"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TimeDeltaException(Exception):
    def __init__(self, failure_count: int, reset_timeout_sec: float):
        self.failure_count = failure_count
        self.reset_timeout_sec = reset_timeout_sec
        self.message = f"{self.failure_count} failures in {self.reset_timeout_sec} seconds"
        super().__init__(self.message)


def unpack_pose_message(packed_data: bytes, topic: str = "pose") -> dict:
    """Unpack a single-frame packed message from pico_manager_thread_server.

    Wire format: [topic_prefix][1280-byte JSON header][concatenated binary fields]
    """
    HEADER_SIZE = 1280

    topic_bytes = topic.encode("utf-8")
    if not packed_data.startswith(topic_bytes):
        raise ValueError(f"Message does not start with expected topic '{topic}'")

    offset = len(topic_bytes)
    if len(packed_data) < offset + HEADER_SIZE:
        raise ValueError(f"Packed data too small: {len(packed_data)} < {offset + HEADER_SIZE}")

    header_bytes = packed_data[offset : offset + HEADER_SIZE]
    null_idx = header_bytes.find(b"\x00")
    if null_idx > 0:
        header_bytes = header_bytes[:null_idx]

    header = json.loads(header_bytes.decode("utf-8"))
    fields = header.get("fields", [])

    result = {"version": header.get("v", 0), "endian": header.get("endian", "le")}
    current_offset = offset + HEADER_SIZE
    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "bool": bool,
    }

    for field in fields:
        dtype = dtype_map.get(field["dtype"], np.float32)
        shape = tuple(field["shape"])
        n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        result[field["name"]] = (
            np.frombuffer(packed_data[current_offset : current_offset + n_bytes], dtype=dtype)
            .reshape(shape)
            .copy()
        )
        current_offset += n_bytes

    return result


class TimingThresholdMonitor:
    def __init__(self, max_failures=3, reset_timeout_sec=5, time_delta=0.2, raise_exception=False):
        self.max_failures = max_failures
        self.reset_timeout_sec = reset_timeout_sec
        self.failure_count = 0
        self.last_failure_time = 0
        self.time_delta = time_delta
        self.raise_exception = raise_exception

    def reset(self):
        self.failure_count = 0
        self.last_failure_time = 0

    def log_time_delta(self, time_delta_sec: float):
        time_delta = abs(time_delta_sec)
        if time_delta > self.time_delta:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()

        if self.is_threshold_exceeded():
            print(
                f"Time delta exception: {self.failure_count} failures in "
                f"{self.reset_timeout_sec} seconds, time delta: {time_delta}"
            )
            if self.raise_exception:
                raise TimeDeltaException(self.failure_count, self.reset_timeout_sec)

    def is_threshold_exceeded(self):
        if self.failure_count >= self.max_failures:
            return True
        if time.monotonic() - self.last_failure_time > self.reset_timeout_sec:
            self.reset()
        return False


# ---------------------------------------------------------------------------
# Data Collector
# ---------------------------------------------------------------------------


class GrootDataCollector:
    """Collects data from G1 robot in Sonic CPP + SMPL mode -- no ROS 2.

    Data sources (all ZMQ):
      - ``g1_debug`` topic        -> proprio (body_q, hand_q, actions, base_quat, ...)
      - ``pose`` topic            -> SMPL pose (smpl_joints, body_quat_w, hand_joints, ...)
      - ``planner`` topic         -> planner commands (vr_position, vr_orientation, ...)
      - ``manager_state`` topic   -> current stream mode + toggle flags
      - Camera client             -> ego-view images
    """

    def __init__(
        self,
        camera_host: str,
        camera_port: int,
        data_exporter: Gr00tDataExporter,
        robot_model,
        text_to_speech=None,
        frequency: int = 20,
        sonic_data_zmq_host: str = "localhost",
        sonic_data_zmq_port: int = 5556,
        state_zmq_host: str = "localhost",
        state_zmq_port: int = 5557,
        keyboard_port: int = 5580,
        ignore_manager_toggles: bool = False,
        eef_zmq_port: int = 0,
        frame_clock: bool = False,
        lean: bool = False,
    ):
        self.text_to_speech = text_to_speech
        self._frame_clock = frame_clock
        self._lean = lean
        self._cmd_time = None  # boundary time carried by the last start/stop command (frame clock)
        self._take_cmd = False  # the episode being started came from a take command (frame clock)
        self._fc = None  # frame clock: the open episode's emission state
        self._ticks: dict[int, _Tick] = {}  # frame clock: the last ~1 s of ticks, by tick index
        self._k = 0  # frame clock: the newest tick
        self.frequency = frequency
        self.loop_period = 1.0 / frequency
        self.data_exporter = data_exporter
        self.robot_model = robot_model

        self._episode_state = EpisodeState()
        self._keyboard_listener = ZMQKeyboardSubscriber(port=keyboard_port)
        self._ignore_manager_toggles = ignore_manager_toggles
        self._take_id = ""
        self._take_t0 = None
        self._takes_log = f"{self.data_exporter.meta.root}/coscene_takes.jsonl"

        self._image_subscriber = ComposedCameraClientSensor(server_ip=camera_host, port=camera_port)

        self.obs_act_buffer = deque(maxlen=100)
        self.latest_image_msg = None
        self.latest_proprio_msg = None
        self.latest_sonic_msg = None
        self.latest_planner_msg = None

        self.current_stream_mode = 0

        self._manager_toggle_dc = False
        self._manager_toggle_da = False

        self._state_subscriber = ZMQStateSubscriber(
            host=state_zmq_host,
            port=state_zmq_port,
        )

        self._sonic_zmq_ctx = None
        self._sonic_zmq_socket = None
        self._manager_state_socket = None
        self._eef_socket = None
        self._latest_eef = None
        if eef_zmq_port > 0:
            self._eef_ctx = zmq.Context()
            self._eef_socket = self._eef_ctx.socket(zmq.SUB)
            self._eef_socket.setsockopt(zmq.CONFLATE, 1)
            self._eef_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._eef_socket.connect(f"tcp://localhost:{eef_zmq_port}")
            print(f"[Gripper] Subscribed to eef_zmq_tap on port {eef_zmq_port}")
        try:
            self._sonic_zmq_ctx = zmq.Context()
            self._sonic_zmq_socket = self._sonic_zmq_ctx.socket(zmq.SUB)
            self._sonic_zmq_socket.connect(f"tcp://{sonic_data_zmq_host}:{sonic_data_zmq_port}")
            self._sonic_zmq_socket.setsockopt(zmq.RCVTIMEO, 100)
            self._sonic_zmq_socket.setsockopt(zmq.CONFLATE, 0)
            self._sonic_zmq_socket.setsockopt(zmq.RCVHWM, 20)
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "pose")
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "planner")
            if ignore_manager_toggles:
                # own socket, latest message only: stream_mode is a level, and the toggles are ignored
                self._manager_state_socket = self._sonic_zmq_ctx.socket(zmq.SUB)
                self._manager_state_socket.setsockopt(zmq.CONFLATE, 1)
                self._manager_state_socket.setsockopt(zmq.RCVTIMEO, 0)
                self._manager_state_socket.setsockopt_string(zmq.SUBSCRIBE, "manager_state")
                self._manager_state_socket.connect(f"tcp://{sonic_data_zmq_host}:{sonic_data_zmq_port}")
            else:
                self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "manager_state")
            time.sleep(0.5)
            print(f"[Sonic] Connected to ZMQ at {sonic_data_zmq_host}:{sonic_data_zmq_port}")
            print("[Sonic] Subscribed to: pose, planner, manager_state")
        except Exception as e:
            print(f"[Sonic] Warning: Failed to initialize ZMQ subscriber: {e}")
            self._sonic_zmq_socket = None

        self.telemetry = Telemetry(window_size=100)
        self.sonic_timing_monitor = TimingThresholdMonitor(
            max_failures=3, reset_timeout_sec=5, time_delta=0.1
        )

        self._last_latency_log_time = 0.0
        self._initial_yaw = None

        print(f"Recording to {self.data_exporter.meta.root}")

    @property
    def current_episode_index(self):
        return self.data_exporter.episode_buffer["episode_index"]

    def _print_and_say(self, message: str, say: bool = True, blocking: bool = False):
        if self.text_to_speech is not None:
            self.text_to_speech.print_and_say(message, say, blocking=blocking)
        else:
            print(message)

    def _poll_state_zmq(self):
        """Poll the ``g1_debug`` ZMQ topic for robot state (non-blocking)."""
        msg = self._state_subscriber.get_msg(clear=True)
        if msg is None:
            return

        if msg.get("ros_timestamp", 0.0) == 0.0:
            msg["ros_timestamp"] = time.time()

        self.latest_proprio_msg = msg

    def _check_recording_commands(self):
        """Check keyboard + ZMQ toggle flags for recording commands."""
        key = self._keyboard_listener.read_msg()
        if key and " " in key or key in ("start", "stop", "discard"):
            self._handle_take_command(key)
            key = None

        if self._ignore_manager_toggles:
            self._manager_toggle_da = self._manager_toggle_dc = False
        if self._manager_toggle_da:
            key = "x"
            self._manager_toggle_da = False
        elif self._manager_toggle_dc:
            key = "c"
            self._manager_toggle_dc = False

        if key == "c":
            self._episode_state.change_state()
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                self._initial_yaw = None
                self._print_and_say(
                    f"Started recording {self.current_episode_index}", blocking=False
                )
            elif self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
                self._print_and_say("Stopping recording, preparing to save", blocking=False)
            elif self._episode_state.get_state() == self._episode_state.IDLE:
                self._print_and_say("Saved episode and back to idle state", blocking=False)
        elif key == "x":
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                self._discard_episode()
                self._print_and_say("Discarded episode", blocking=False)

    def _discard_episode(self) -> None:
        if self._fc is not None:  # frame clock: the rows still behind the lag first, so the episode is not empty
            self._fc_emit(self._k)
            self._fc = None
        self.data_exporter.save_episode_as_discarded()
        self._episode_state.reset_state()
        self._initial_yaw = None

    def _handle_take_command(self, msg: str) -> None:
        """Idempotent start/stop/discard from pico_record.py ('start <take id> [<unix time>]' ...); repeats are
        no-ops. The optional time is the boundary the sender asks for (its coScene request time); only the frame
        clock uses it."""
        cmd, _, take_id = msg.partition(" ")
        head, _, tail = take_id.rpartition(" ")
        try:
            t_cmd = float(tail) if head else None
        except ValueError:
            t_cmd = None
        if t_cmd is not None:
            take_id = head
        state = self._episode_state.get_state()
        if cmd == "start" and state == self._episode_state.IDLE:
            self._episode_state.change_state()
            self._initial_yaw = None
            self._take_id, self._take_t0 = take_id, time.time()
            self._cmd_time, self._take_cmd = t_cmd, True
            print(f"[Take] start episode {self.current_episode_index} take {take_id!r} at {self._take_t0:.6f}"
                  + (f" (requested {t_cmd:.6f})" if t_cmd is not None else ""))
        elif cmd == "stop" and state == self._episode_state.RECORDING:
            self._episode_state.change_state()
            t1 = time.time()
            self._cmd_time = t_cmd
            n = self.data_exporter.episode_buffer.get("size", 0)
            print(f"[Take] stop episode {self.current_episode_index} take {take_id!r} at {t1:.6f}"
                  + (f" (requested {t_cmd:.6f})" if t_cmd is not None else f", {n} frames"))
            if not self._frame_clock:  # the frame clock writes it once the last row is emitted
                self._log_take({"stop_time": t1, "frames": int(n)})
        elif cmd == "discard" and state == self._episode_state.RECORDING:
            self._discard_episode()
            print(f"[Take] discarded take {take_id!r}")

    def _log_take(self, fields: dict) -> None:
        try:
            with open(self._takes_log, "a") as f:
                f.write(json.dumps({"episode_index": int(self.current_episode_index), "take_id": self._take_id,
                                    "start_time": self._take_t0, **fields}) + "\n")
        except OSError as e:
            print(f"[Take] could not append {self._takes_log}: {e}")

    def _poll_eef(self) -> None:
        if self._eef_socket is None:
            return
        try:
            self._latest_eef = json.loads(self._eef_socket.recv_string(zmq.NOBLOCK))
        except zmq.Again:
            pass

    def _add_gripper_features(self, frame_data: dict, now: float | None = None) -> None:
        if self._eef_socket is None:
            return
        now = time.time() if now is None else now
        obs = np.full(4, np.nan, dtype=np.float32)
        act = np.full(2, np.nan, dtype=np.float32)
        msg = self._latest_eef
        if msg is not None:
            for i, side in enumerate(("left", "right")):
                d = msg.get(side) or {}
                if d.get("q") is not None and now - d["t_state"] < 0.5:
                    obs[i] = d["q"]
                if d.get("opening_mm") is not None and now - d["t_opening"] < 0.5:
                    obs[2 + i] = d["opening_mm"]
                if d.get("cmd_q") is not None and now - d["t_cmd"] < 0.5:
                    act[i] = d["cmd_q"]
        frame_data["observation.gripper"] = obs
        frame_data["action.gripper"] = act

    def _poll_sonic_zmq_messages(self):
        """Poll ZMQ for pose, planner, and manager_state messages (non-blocking)."""
        if self._sonic_zmq_socket is None:
            return

        if self._manager_state_socket is not None:
            try:
                self._handle_manager_state(self._manager_state_socket.recv(zmq.NOBLOCK))
            except zmq.Again:
                pass

        max_polls = 20
        for _ in range(max_polls):
            try:
                raw = self._sonic_zmq_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

            if raw.startswith(b"manager_state"):
                self._handle_manager_state(raw)
            elif raw.startswith(b"planner"):
                self._handle_planner_message(raw)
            elif raw.startswith(b"pose"):
                self._handle_pose_message(raw)

    def _handle_manager_state(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, topic="manager_state")
        except Exception:
            return

        if "stream_mode" in data:
            self.current_stream_mode = int(data["stream_mode"].flat[0])

        if self._extract_bool(data, "toggle_data_collection"):
            self._manager_toggle_dc = True
        if self._extract_bool(data, "toggle_data_abort"):
            self._manager_toggle_da = True

    def _handle_planner_message(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, topic="planner")
        except Exception:
            return

        planner_mode = int(data["mode"].flat[0]) if "mode" in data else 0
        planner_movement = (
            data["movement"].flatten().astype(np.float32)
            if "movement" in data and data["movement"].size == 3
            else np.zeros(3, dtype=np.float32)
        )
        planner_facing = (
            data["facing"].flatten().astype(np.float32)
            if "facing" in data and data["facing"].size == 3
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        planner_speed = float(data["speed"].flat[0]) if "speed" in data else -1.0
        planner_height = float(data["height"].flat[0]) if "height" in data else -1.0

        vr_3pt_position = None
        if "vr_position" in data and data["vr_position"].size == 9:
            vr_3pt_position = data["vr_position"].flatten().astype(np.float32)
        vr_3pt_orientation = None
        if "vr_orientation" in data and data["vr_orientation"].size == 12:
            vr_3pt_orientation = data["vr_orientation"].flatten().astype(np.float32)

        self.latest_planner_msg = {
            "planner_mode": planner_mode,
            "planner_movement": planner_movement,
            "planner_facing": planner_facing,
            "planner_speed": planner_speed,
            "planner_height": planner_height,
            "vr_3pt_position": vr_3pt_position,
            "vr_3pt_orientation": vr_3pt_orientation,
            "left_hand_joints": self._extract_hand_joints(data, "left_hand_joints"),
            "right_hand_joints": self._extract_hand_joints(data, "right_hand_joints"),
            "receive_timestamp": time.time(),
        }

    def _handle_pose_message(self, raw: bytes) -> None:
        G1_L_WRIST_ROLL_IDX = 23
        G1_L_WRIST_PITCH_IDX = 25
        G1_L_WRIST_YAW_IDX = 27
        G1_R_WRIST_ROLL_IDX = 24
        G1_R_WRIST_PITCH_IDX = 26
        G1_R_WRIST_YAW_IDX = 28

        try:
            pose_data = unpack_pose_message(raw, topic="pose")
        except Exception as e:
            print(f"[Sonic] Error unpacking pose message: {e}")
            return

        try:
            if "smpl_joints" not in pose_data or len(pose_data["smpl_joints"].shape) != 3:
                return

            left_wrist_joints = None
            right_wrist_joints = None
            if "joint_pos" in pose_data and len(pose_data["joint_pos"].shape) == 2:
                joint_pos = pose_data["joint_pos"][0]
                left_wrist_joints = np.array(
                    [
                        joint_pos[G1_L_WRIST_ROLL_IDX],
                        joint_pos[G1_L_WRIST_PITCH_IDX],
                        joint_pos[G1_L_WRIST_YAW_IDX],
                    ],
                    dtype=np.float32,
                )
                right_wrist_joints = np.array(
                    [
                        joint_pos[G1_R_WRIST_ROLL_IDX],
                        joint_pos[G1_R_WRIST_PITCH_IDX],
                        joint_pos[G1_R_WRIST_YAW_IDX],
                    ],
                    dtype=np.float32,
                )

            frame_index = None
            if "frame_index" in pose_data:
                frame_index = np.array([pose_data["frame_index"].flat[0]], dtype=np.int64)

            smpl_pose = np.zeros(63, dtype=np.float32)
            if "smpl_pose" in pose_data:
                raw_pose = pose_data["smpl_pose"]
                if raw_pose.ndim == 3:
                    smpl_pose = raw_pose[0].flatten().astype(np.float32)
                elif raw_pose.ndim == 2:
                    smpl_pose = raw_pose.flatten().astype(np.float32)
                elif raw_pose.ndim == 1 and raw_pose.size == 63:
                    smpl_pose = raw_pose.astype(np.float32)

            left_hand_joints = self._extract_hand_joints(pose_data, "left_hand_joints")
            right_hand_joints = self._extract_hand_joints(pose_data, "right_hand_joints")

            vr_3pt_position = None
            if "vr_position" in pose_data and pose_data["vr_position"].size == 9:
                vr_3pt_position = pose_data["vr_position"].flatten().astype(np.float32)
            vr_3pt_orientation = None
            if "vr_orientation" in pose_data and pose_data["vr_orientation"].size == 12:
                vr_3pt_orientation = pose_data["vr_orientation"].flatten().astype(np.float32)

            self.latest_sonic_msg = {
                "smpl_joints": pose_data["smpl_joints"][0],
                "smpl_pose": smpl_pose,
                "body_quat_w": (
                    pose_data["body_quat_w"][0] if "body_quat_w" in pose_data else None
                ),
                "left_hand_joints": left_hand_joints,
                "right_hand_joints": right_hand_joints,
                "left_wrist_joints": left_wrist_joints,
                "right_wrist_joints": right_wrist_joints,
                "vr_3pt_position": vr_3pt_position,
                "vr_3pt_orientation": vr_3pt_orientation,
                "frame_index": frame_index,
                "receive_timestamp": time.time(),
            }
        except Exception as e:
            if not hasattr(self, "_sonic_error_count"):
                self._sonic_error_count = 0
            self._sonic_error_count += 1
            if self._sonic_error_count == 1 or self._sonic_error_count % 100 == 0:
                print(f"[Sonic] Error processing pose message: {e}")

    @staticmethod
    def _extract_hand_joints(pose_data: dict, key: str) -> np.ndarray:
        arr = pose_data.get(key)
        if arr is not None:
            if arr.ndim > 1:
                arr = arr[0]
            return arr.astype(np.float32)
        return np.zeros(7, dtype=np.float32)

    @staticmethod
    def _extract_bool(pose_data: dict, key: str) -> bool:
        val = pose_data.get(key)
        if val is None:
            return False
        if isinstance(val, np.ndarray):
            return bool(val.flat[0])
        return bool(val)

    def _log_latency_periodic(
        self,
        sonic_latency_ms: float | None = None,
    ):
        current_time = time.time()
        if current_time - self._last_latency_log_time >= 1.0:
            self._last_latency_log_time = current_time
            parts = []
            if sonic_latency_ms is not None:
                parts.append(f"Sonic Pose: {sonic_latency_ms:.1f}ms")
            if parts:
                print(f"[Latency] {', '.join(parts)}")

    def _add_images_to_frame_data(self, frame_data: dict) -> None:
        if self.latest_image_msg is None:
            return
        images = self.latest_image_msg["images"]
        for feature_name, feature_info in self.data_exporter.features.items():
            if feature_info.get("dtype") in ["image", "video"]:
                image_key = feature_name.split(".")[-1]
                if image_key not in images:
                    raise ValueError(
                        f"Required image '{image_key}' for feature '{feature_name}' "
                        f"not found in image message. Available: {list(images.keys())}"
                    )
                frame_data[feature_name] = images[image_key]

    def _finalize_frame(self, t_start: float) -> bool:
        t_end = time.monotonic()
        if t_end - t_start > (1 / self.frequency):
            print(f"DataExporter Missed: {t_end - t_start} sec")

        if self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode()
                self.sonic_timing_monitor.reset()
                self._initial_yaw = None
                self._print_and_say("Finished saving episode")
            else:
                self._print_and_say("Skipping save: no frames collected", say=False)
            self._episode_state.change_state()
        return True

    def _add_data_frame(self):
        t_start = time.monotonic()

        if self.latest_proprio_msg is None or self.latest_image_msg is None:
            self._print_and_say(
                f"Waiting for message. "
                f"Avail msg: proprio {self.latest_proprio_msg is not None} | "
                f"image {self.latest_image_msg is not None}",
                say=False,
            )
            return False

        if self._episode_state.get_state() != self._episode_state.RECORDING:
            return self._finalize_frame(t_start)

        return self._add_data_frame_sonic(t_start)

    def _add_data_frame_sonic(self, t_start: float) -> bool:
        """Build one data frame in Sonic CPP + SMPL mode."""
        frame_data, sonic_latency_ms = self._build_frame()
        self._log_latency_periodic(sonic_latency_ms)
        self.data_exporter.add_frame(frame_data)
        return self._finalize_frame(t_start)

    def _build_frame(self, now: float | None = None) -> tuple[dict, float | None]:
        """One row from the latest inputs (self.latest_*), staleness measured at ``now`` (default: now)."""
        assert self.latest_proprio_msg is not None
        proprio = self.latest_proprio_msg

        whole_q = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=proprio["body_q"],
            left_hand_actuated_joint_values=proprio["left_hand_q"],
            right_hand_actuated_joint_values=proprio["right_hand_q"],
        )
        whole_action_wbc = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=proprio["last_action"],
            left_hand_actuated_joint_values=proprio["last_left_hand_action"],
            right_hand_actuated_joint_values=proprio["last_right_hand_action"],
        )

        self.robot_model.cache_forward_kinematics(whole_q)
        eef_parts = []
        for side in ["left", "right"]:
            placement = self.robot_model.frame_placement(
                self.robot_model.supplemental_info.hand_frame_names[side]
            )
            pos = placement.translation[:3]
            quat = R.from_matrix(placement.rotation).as_quat(scalar_first=True)
            eef_parts.append(np.concatenate([pos, quat]))
        observation_eef_state = np.concatenate(eef_parts)

        frame_data: dict = {
            "observation.state": whole_q,
            "observation.eef_state": observation_eef_state,
            "action.wbc": whole_action_wbc,
        }

        self._add_cpp_state_features(frame_data, proprio)

        sonic_latency_ms = self._add_sonic_pose_features(frame_data, now)

        self._add_images_to_frame_data(frame_data)

        self._add_gripper_features(frame_data, now)

        return frame_data, sonic_latency_ms

    def _add_cpp_state_features(self, frame_data: dict, proprio: dict) -> None:
        if "base_quat" in proprio:
            base_quat = np.asarray(proprio["base_quat"], dtype=np.float64)
            frame_data["observation.root_orientation"] = base_quat
            frame_data["observation.projected_gravity"] = compute_projected_gravity(
                base_quat
            ).astype(np.float64)

            if "init_ref_data_root_rot_array" in proprio:
                frame_data["observation.cpp_rotation_offset"] = np.asarray(
                    proprio["init_ref_data_root_rot_array"], dtype=np.float64
                )
            else:
                frame_data["observation.cpp_rotation_offset"] = np.array(
                    [1.0, 0.0, 0.0, 0.0], dtype=np.float64
                )
        else:
            frame_data["observation.root_orientation"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
            frame_data["observation.projected_gravity"] = np.array(
                [0.0, 0.0, -1.0], dtype=np.float64
            )
            frame_data["observation.cpp_rotation_offset"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

        if "init_base_quat" in proprio:
            frame_data["observation.init_base_quat"] = np.asarray(
                proprio["init_base_quat"], dtype=np.float64
            )
        else:
            frame_data["observation.init_base_quat"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

        if "delta_heading" in proprio:
            dh = proprio["delta_heading"]
            if isinstance(dh, np.ndarray):
                dh = dh.item() if dh.size == 1 else dh[0]
            frame_data["teleop.delta_heading"] = np.array([float(dh)], dtype=np.float64)
        else:
            frame_data["teleop.delta_heading"] = np.zeros(1, dtype=np.float64)

        if "token_state" in proprio:
            frame_data["action.motion_token"] = np.asarray(proprio["token_state"], dtype=np.float64)
        else:
            frame_data["action.motion_token"] = np.zeros(64, dtype=np.float64)

    def _add_sonic_pose_features(self, frame_data: dict, now: float | None = None) -> float | None:
        """Add teleop features based on current stream mode (staleness measured at ``now``, default: now)."""
        sonic_latency_ms = None
        now = time.time() if now is None else now

        frame_data["teleop.stream_mode"] = np.array([self.current_stream_mode], dtype=np.int32)

        smpl_msg = self.latest_sonic_msg
        use_smpl = False
        if self.current_stream_mode in (1, 4) and smpl_msg is not None:
            receive_ts = smpl_msg.get("receive_timestamp")
            if receive_ts is not None:
                age_sec = now - receive_ts
                sonic_latency_ms = age_sec * 1000
                self.sonic_timing_monitor.log_time_delta(age_sec)
                if sonic_latency_ms <= 100.0:
                    use_smpl = True
                elif (self.sonic_timing_monitor.failure_count + 1) % 10 == 0:
                    self._print_and_say(
                        f"Sonic pose stale ({sonic_latency_ms:.1f}ms old), using zeros",
                        say=False,
                    )
            else:
                use_smpl = True

        planner_msg = self.latest_planner_msg
        use_planner = False
        if self.current_stream_mode == 5 and planner_msg is not None:
            receive_ts = planner_msg.get("receive_timestamp")
            if receive_ts is not None:
                age_sec = now - receive_ts
                planner_latency_ms = age_sec * 1000
                if sonic_latency_ms is None:
                    sonic_latency_ms = planner_latency_ms
                if planner_latency_ms <= 200.0:
                    use_planner = True
            else:
                use_planner = True

        # SMPL features
        if use_smpl and smpl_msg.get("smpl_joints") is not None:
            joints = np.asarray(smpl_msg["smpl_joints"], dtype=np.float32)
            if joints.ndim == 2:
                joints = joints.flatten()
            frame_data["teleop.smpl_joints"] = np.ascontiguousarray(joints, dtype=np.float32)
        else:
            frame_data["teleop.smpl_joints"] = np.zeros(72, dtype=np.float32)

        if use_smpl and smpl_msg.get("smpl_pose") is not None:
            pose = np.asarray(smpl_msg["smpl_pose"], dtype=np.float32)
            if pose.ndim > 1:
                pose = pose.flatten()
            frame_data["teleop.smpl_pose"] = np.ascontiguousarray(pose, dtype=np.float32)
        else:
            frame_data["teleop.smpl_pose"] = np.zeros(63, dtype=np.float32)

        if use_smpl and smpl_msg.get("body_quat_w") is not None:
            body_quat_w = smpl_msg["body_quat_w"].astype(np.float32)
            frame_data["teleop.body_quat_w"] = body_quat_w
            frame_data["teleop.target_body_orientation"] = self._compute_target_body_orientation(
                body_quat_w, frame_data
            )
        else:
            frame_data["teleop.body_quat_w"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            frame_data["teleop.target_body_orientation"] = quat_to_rot6d(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            )

        frame_data["teleop.left_wrist_joints"] = (
            smpl_msg["left_wrist_joints"].astype(np.float32)
            if use_smpl and smpl_msg.get("left_wrist_joints") is not None
            else np.zeros(3, dtype=np.float32)
        )
        frame_data["teleop.right_wrist_joints"] = (
            smpl_msg["right_wrist_joints"].astype(np.float32)
            if use_smpl and smpl_msg.get("right_wrist_joints") is not None
            else np.zeros(3, dtype=np.float32)
        )

        frame_data["teleop.smpl_frame_index"] = (
            smpl_msg["frame_index"].astype(np.int64)
            if use_smpl and smpl_msg is not None and smpl_msg.get("frame_index") is not None
            else np.array([0], dtype=np.int64)
        )

        hand_msg = (
            smpl_msg if self.current_stream_mode in (1, 4) and smpl_msg is not None
            else planner_msg if planner_msg is not None
            else smpl_msg
        )
        frame_data["teleop.left_hand_joints"] = (
            hand_msg["left_hand_joints"].astype(np.float32)
            if hand_msg is not None
            and hand_msg.get("left_hand_joints") is not None
            else np.zeros(7, dtype=np.float32)
        )
        frame_data["teleop.right_hand_joints"] = (
            hand_msg["right_hand_joints"].astype(np.float32)
            if hand_msg is not None
            and hand_msg.get("right_hand_joints") is not None
            else np.zeros(7, dtype=np.float32)
        )

        # Planner command fields
        frame_data["teleop.planner_mode"] = np.array(
            [planner_msg["planner_mode"]] if use_planner else [0],
            dtype=np.int32,
        )
        frame_data["teleop.planner_movement"] = (
            planner_msg["planner_movement"].copy()
            if use_planner and planner_msg.get("planner_movement") is not None
            else np.zeros(3, dtype=np.float32)
        )
        frame_data["teleop.planner_facing"] = (
            planner_msg["planner_facing"].copy()
            if use_planner and planner_msg.get("planner_facing") is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        frame_data["teleop.planner_speed"] = np.array(
            [planner_msg["planner_speed"]] if use_planner else [-1.0],
            dtype=np.float32,
        )
        frame_data["teleop.planner_height"] = np.array(
            [planner_msg["planner_height"]] if use_planner else [-1.0],
            dtype=np.float32,
        )

        # VR 3-point pose
        frame_data["teleop.vr_3pt_position"] = (
            planner_msg["vr_3pt_position"].astype(np.float32)
            if use_planner and planner_msg.get("vr_3pt_position") is not None
            else np.zeros(9, dtype=np.float32)
        )
        if use_planner and planner_msg.get("vr_3pt_orientation") is not None:
            frame_data["teleop.vr_3pt_orientation"] = quat_to_rot6d(
                planner_msg["vr_3pt_orientation"].astype(np.float32)
            )
        else:
            frame_data["teleop.vr_3pt_orientation"] = np.zeros(18, dtype=np.float32)

        return sonic_latency_ms

    def _compute_target_body_orientation(
        self, body_quat_w: np.ndarray, frame_data: dict
    ) -> np.ndarray:
        """Compute yaw-normalised target body orientation as rot6d (6-dim)."""
        delta_heading = float(frame_data.get("teleop.delta_heading", [0.0])[0])

        body_rot = R.from_quat(body_quat_w, scalar_first=True)
        target_rot = R.from_euler("z", delta_heading, degrees=False) * body_rot

        euler = target_rot.as_euler("ZYX", degrees=False)
        current_yaw = euler[0]

        if self._initial_yaw is None:
            self._initial_yaw = current_yaw

        normalised_euler = np.array([current_yaw - self._initial_yaw, euler[1], euler[2]])
        target_quat = (
            R.from_euler("ZYX", normalised_euler, degrees=False)
            .as_quat(scalar_first=True)
            .astype(np.float32)
        )
        return quat_to_rot6d(target_quat)

    def save_and_cleanup(self):
        try:
            self._print_and_say("saving episode done", blocking=False)
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode()
            self._print_and_say(
                f"Recording complete: {self.data_exporter.meta.root}", say=False, blocking=True
            )
        except Exception as e:
            self._print_and_say(f"Error saving episode: {e}", blocking=True)

        try:
            self._state_subscriber.close()
        except Exception:
            pass
        for sock in [self._sonic_zmq_socket, self._manager_state_socket, self._eef_socket]:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        for ctx in [self._sonic_zmq_ctx]:
            if ctx is not None:
                try:
                    ctx.term()
                except Exception:
                    pass

        self._print_and_say("Shutting down data exporter...", say=False)

    def _poll_image(self) -> None:
        if self._lean:  # keep the raw message; LazyImages decodes what a written row needs
            msg = self._image_subscriber.receive_message_nonblocking(timeout_ms=0)
            if msg is not None:
                self.latest_image_msg = {"timestamps": msg.get("timestamps", {}),
                                         "images": LazyImages(msg.get("images", {}))}
            return
        img_msg = self._image_subscriber.read()
        if img_msg is not None:
            self.latest_image_msg = img_msg

    # ------------------------------------------------------------------ frame clock (--frame-clock)

    @staticmethod
    def _tick_ready(t: "_Tick | None") -> bool:
        return t is not None and t.proprio is not None and t.image is not None

    def _fc_begin(self, k: int, anchor: float) -> None:
        """Open an episode at the tick nearest the requested start time (default: this tick)."""
        period, t = self.loop_period, self._cmd_time
        want = k if t is None else round((t - anchor) / period)
        j = min(max(want, next(iter(self._ticks))), k)
        if j != want:
            print(f"[FrameClock] start time {t:.6f} is {(k - want) * period:.3f} s before this tick; "
                  f"the pre-roll reaches {(k - j) * period:.3f} s")
        self._fc = {"next": j, "stop": None, "t0": anchor + j * period, "rows": 0, "held": 0, "prev": None,
                    "take": self._take_cmd, "start_request": t, "stop_request": None, "stop_seen": None}
        self._take_cmd = False

    def _fc_end(self, k: int, anchor: float) -> None:
        """Close the episode before the tick nearest the requested stop time (default: after this tick)."""
        fc, period, t = self._fc, self.loop_period, self._cmd_time
        want = k + 1 if t is None else round((t - anchor) / period)
        fc["stop"] = min(max(want, fc["next"]), k + 1)
        fc["stop_request"], fc["stop_seen"] = t, time.time()
        if fc["stop"] != want:
            print(f"[FrameClock] stop time {t:.6f}: rows up to tick {fc['stop'] - 1} were already written "
                  f"({fc['stop'] - want} past the requested stop)")

    def _fc_emit(self, upto: int) -> None:
        """Write the rows of ticks fc['next'] .. upto. A tick the loop missed repeats the previous row (held)."""
        fc = self._fc
        while fc["next"] <= upto:
            j = fc["next"]
            fc["next"] += 1
            tick = self._ticks.get(j)
            held = not self._tick_ready(tick)
            if held:
                tick = fc["prev"] or next(
                    (t for t in reversed(self._ticks.values()) if t.k < j and self._tick_ready(t)), None)
            if not self._tick_ready(tick):  # no input yet: the episode starts at a later tick
                if fc["rows"] == 0:
                    fc["t0"] += self.loop_period
                continue
            self._fc_add_row(tick, held)
            fc["prev"] = tick
            fc["rows"] += 1
            fc["held"] += held

    def _fc_add_row(self, tick: "_Tick", held: bool) -> None:
        live = (self.latest_proprio_msg, self.latest_sonic_msg, self.latest_planner_msg,
                self.current_stream_mode, self.latest_image_msg, self._latest_eef)
        (self.latest_proprio_msg, self.latest_sonic_msg, self.latest_planner_msg,
         self.current_stream_mode, self.latest_image_msg, self._latest_eef) = (
            tick.proprio, tick.sonic, tick.planner, tick.stream_mode, tick.image, tick.eef)
        try:
            frame, sonic_latency_ms = self._build_frame(now=tick.wall)
        finally:
            (self.latest_proprio_msg, self.latest_sonic_msg, self.latest_planner_msg,
             self.current_stream_mode, self.latest_image_msg, self._latest_eef) = live
        frame["wall_time"] = np.array([tick.wall], dtype=np.float64)
        frame["held"] = np.array([int(held)], dtype=np.int32)
        frame["state_time"] = np.array([float(tick.proprio.get("ros_timestamp", np.nan))], dtype=np.float64)
        frame["camera_time"] = np.array(
            [float(tick.image.get("timestamps", {}).get(self._image_time_key, np.nan))], dtype=np.float64)
        if not held:
            self._log_latency_periodic(sonic_latency_ms)
        self.data_exporter.add_frame(frame)

    def _fc_finish(self) -> None:
        fc, period = self._fc, self.loop_period
        n = int(self.data_exporter.episode_buffer.get("size", 0))
        print(f"[FrameClock] episode {self.current_episode_index}: {n} rows ({fc['held']} held), "
              f"t0_wall {fc['t0']:.6f}, rows end {fc['t0'] + n * period:.6f}")
        if fc["take"]:
            self._log_take({"stop_time": fc["stop_seen"], "frames": n, "t0_wall": fc["t0"],
                            "held_rows": fc["held"], "start_request": fc["start_request"],
                            "stop_request": fc["stop_request"]})
        self._fc = None
        self._finalize_frame(time.monotonic())  # saves and goes back to idle

    def _run_frame_clock(self):
        """Ticks on absolute deadlines (tick k at t0 + k/fps, monotonic). Each tick reads the inputs once and
        keeps references to them (_Tick) for ~1 s. Rows are written from the ticks `lag` ticks behind the newest
        one, so a stop time that arrives after it still cuts exactly, and a start time from up to 1 s ago opens
        the episode at its own tick (pre-roll). A tick the loop missed becomes a held row: rows stay one per tick,
        so timestamp = frame_index / fps keeps matching the wall clock."""
        period = self.loop_period
        lag = max(2, round(0.1 * self.frequency))
        keep = self.frequency + lag + 1
        self._image_time_key = next((name.split(".")[-1] for name, ft in self.data_exporter.features.items()
                                     if ft.get("dtype") in ("image", "video")), None)
        state = self._episode_state
        mono0 = time.monotonic()
        k_next = 0
        print(f"[FrameClock] {self.frequency} Hz, rows {lag} ticks behind, pre-roll {self.frequency} ticks")
        try:
            while True:
                delay = mono0 + k_next * period - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                t_start = time.monotonic()
                with self.telemetry.timer("total_loop"):
                    with self.telemetry.timer("poll_state"):
                        self._poll_state_zmq()
                    with self.telemetry.timer("poll_sonic"):
                        self._poll_sonic_zmq_messages()
                    self._poll_eef()
                    with self.telemetry.timer("poll_image"):
                        self._poll_image()
                    now_mono, now_wall = time.monotonic(), time.time()
                    k = self._k = int((now_mono - mono0) / period)
                    self._ticks[k] = _Tick(k, now_wall, self.latest_proprio_msg, self.latest_sonic_msg,
                                           self.latest_planner_msg, self.current_stream_mode,
                                           self.latest_image_msg, self._latest_eef)
                    while next(iter(self._ticks)) < k - keep:
                        del self._ticks[next(iter(self._ticks))]
                    before = state.get_state()
                    with self.telemetry.timer("check_recording_commands"):
                        self._check_recording_commands()
                    after = state.get_state()
                    if after != before:
                        anchor = now_wall - (now_mono - mono0)  # wall time of tick 0, as of this tick
                        if after == state.RECORDING:
                            self._fc_begin(k, anchor)
                        elif after == state.NEED_TO_SAVE and self._fc is not None:
                            self._fc_end(k, anchor)
                        elif after == state.NEED_TO_SAVE:
                            self._finalize_frame(time.monotonic())
                        self._cmd_time = None
                    with self.telemetry.timer("add_frame"):
                        if self._fc is not None:
                            stop = self._fc["stop"]
                            self._fc_emit(k - lag if stop is None else stop - 1)
                            if stop is not None:
                                self._fc_finish()
                k_next = k + 1
                if time.monotonic() - t_start > period:
                    self.telemetry.log_timing_info(context="Data Exporter Loop Missed", threshold=0.001)

        except KeyboardInterrupt:
            print("Data exporter terminated by user")
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode_as_discarded()

        finally:
            self.save_and_cleanup()

    def run(self):
        if self._frame_clock:
            return self._run_frame_clock()
        try:
            while True:
                t_start = time.monotonic()
                with self.telemetry.timer("total_loop"):
                    with self.telemetry.timer("poll_state"):
                        self._poll_state_zmq()

                    with self.telemetry.timer("poll_sonic"):
                        self._poll_sonic_zmq_messages()

                    self._poll_eef()

                    with self.telemetry.timer("poll_image"):
                        self._poll_image()

                    with self.telemetry.timer("add_frame"):
                        self._add_data_frame()

                    with self.telemetry.timer("check_recording_commands"):
                        self._check_recording_commands()

                    end_time = time.monotonic()

                elapsed = time.monotonic() - t_start
                sleep_time = self.loop_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if (end_time - t_start) > self.loop_period:
                    self.telemetry.log_timing_info(
                        context="Data Exporter Loop Missed", threshold=0.001
                    )

        except KeyboardInterrupt:
            print("Data exporter terminated by user")
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode_as_discarded()

        finally:
            self.save_and_cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config: SonicDataExporterConfig):
    g1_rm = get_g1_robot_model()

    dataset_features = get_features_sonic_vla(g1_rm)
    modality_config = get_modality_config_sonic_vla(g1_rm)

    if config.record_wrist_cameras:
        print("[Camera] Wrist cameras enabled — adding to dataset schema")
        dataset_features.update(get_wrist_camera_features())
        wrist_modality = get_wrist_camera_modality_config()
        for key, value in wrist_modality.items():
            if key in modality_config:
                modality_config[key].update(value)
            else:
                modality_config[key] = value

    if config.eef_zmq_port > 0:
        dataset_features.update(GRIPPER_FEATURES)

    if config.frame_clock:
        dataset_features.update(FRAME_CLOCK_FEATURES)

    video_options = {}
    if config.video_preset:
        video_options["preset"] = config.video_preset
    if config.video_threads > 0:
        video_options["threads"] = str(config.video_threads)

    text_to_speech = TextToSpeech() if config.text_to_speech else None

    robot_config = poll_robot_config_zmq(
        config.state_zmq_host, config.state_zmq_port, config.robot_config_timeout
    )

    exporter_cls = Gr00tDataExporter
    if video_options:

        class _WithVideoOptions(Gr00tDataExporter):
            """Episode videos with the encoder options above (applied when each writer's encoder opens)."""

            def create_video_writer(self):
                writers = super().create_video_writer()
                for writer in writers.values():
                    writer.stream.codec_context.options = dict(video_options)
                return writers

        exporter_cls = _WithVideoOptions

    data_exporter = exporter_cls.create(
        save_root=f"{config.root_output_dir}/{config.dataset_name}",
        fps=config.data_collection_frequency,
        features=dataset_features,
        modality_config=modality_config,
        task=config.task_prompt,
        script_config={**robot_config, "record_wrist_cameras": config.record_wrist_cameras},
    )

    data_collector = GrootDataCollector(
        frequency=config.data_collection_frequency,
        data_exporter=data_exporter,
        robot_model=g1_rm,
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        text_to_speech=text_to_speech,
        sonic_data_zmq_host=config.sonic_zmq_host,
        sonic_data_zmq_port=config.sonic_zmq_port,
        state_zmq_host=config.state_zmq_host,
        state_zmq_port=config.state_zmq_port,
        keyboard_port=config.keyboard_port,
        ignore_manager_toggles=config.ignore_manager_toggles,
        eef_zmq_port=config.eef_zmq_port,
        frame_clock=config.frame_clock,
        lean=config.lean,
    )
    data_collector.run()


if __name__ == "__main__":
    config = tyro.cli(SonicDataExporterConfig)

    if config.dataset_name is None:
        config.dataset_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    main(config)
