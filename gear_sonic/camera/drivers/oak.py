"""OAK (DepthAI) camera driver.

Requires the ``depthai`` SDK — install with::

    pip install depthai

See https://docs.luxonis.com/ for hardware-specific instructions.
"""

import threading
import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

import depthai as dai

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import (
    CameraMountPosition,
    ImageMessageSchema,
    SensorServer,
)


class OAKConfig:
    """Configuration for the OAK camera."""

    color_image_dim: tuple[int, int] = (640, 480)
    monochrome_image_dim: tuple[int, int] = (640, 480)
    fps: int = 30
    enable_color: bool = True
    enable_mono_cameras: bool = False
    mount_position: str = CameraMountPosition.EGO_VIEW.value
    autofocus: bool = False
    manual_focus: int = 130
    use_mjpeg: bool = False
    mjpeg_quality: int = 80
    enable_stereo_stitch: bool = False   # U29: CAM_B+CAM_C synced side-by-side GRAY8, host JPEG, key <mount>_stereo
    enable_depth: bool = False           # U29: on-device StereoDepth, uint16 mm PNG (RECTIFIED_LEFT), key <mount>_depth
    depth_fps_divisor: int = 2           # publish depth every Nth frame (2 -> 15 fps at fps=30)
    stitch_jpeg_quality: int = 80


class OAKSensor(Sensor, SensorServer):
    """Sensor for the OAK camera family (OAK-D, OAK-1, etc.)."""

    def __init__(
        self,
        run_as_server: bool = False,
        port: int = 5555,
        config: OAKConfig = OAKConfig(),
        device_id: str | None = None,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
    ):
        self.config = config
        self.mount_position = mount_position
        self._run_as_server = run_as_server

        device_infos = dai.Device.getAllAvailableDevices()
        assert len(device_infos) > 0, f"No OAK devices found for {mount_position}"
        print(f"Device infos: {device_infos}")
        if device_id is not None:
            device_found = False
            for device_info in device_infos:
                if device_info.getDeviceId() == device_id:
                    self.device = dai.Device(device_info, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)
                    device_found = True
                    break
            if not device_found:
                raise ValueError(f"Device with ID {device_id} not found")
        else:
            self.device = dai.Device()

        print(f"Connected to OAK device: {self.device.getDeviceName(), self.device.getDeviceId()}")
        print(f"Device ID: {self.device.getDeviceId()}")
        try:  # U40 probe: one-shot link/thermal baseline at init
            print(f"[P40] init usb={self.device.getUsbSpeed()} temp={self.device.getChipTemperature().average:.1f}C")
        except Exception as _e:
            print(f"[P40] init probe unavailable: {_e}")

        sockets: list[dai.CameraBoardSocket] = self.device.getConnectedCameras()
        print(f"Available cameras: {[str(s) for s in sockets]}")

        self.pipeline = dai.Pipeline(self.device)
        self.output_queues = {}
        self._use_mjpeg = config.use_mjpeg

        # RGB camera (CAM_A)
        if config.enable_color and dai.CameraBoardSocket.CAM_A in sockets:
            self.cam_rgb = self.pipeline.create(dai.node.Camera)
            self.cam_rgb.initialControl.setAutoExposureLimit(32000)   # U36 09-29: cap AE at 32ms so dim scenes dim the image instead of dropping below 30fps
            cam_socket = dai.CameraBoardSocket.CAM_A
            self.cam_rgb = self.cam_rgb.build(cam_socket)

            if config.use_mjpeg:
                cam_out = self.cam_rgb.requestOutput(
                    config.color_image_dim,
                    dai.ImgFrame.Type.NV12,
                    fps=config.fps,
                )
                encoder = self.pipeline.create(dai.node.VideoEncoder)
                encoder.setDefaultProfilePreset(
                    config.fps, dai.VideoEncoderProperties.Profile.MJPEG
                )
                encoder.setQuality(config.mjpeg_quality)
                cam_out.link(encoder.input)
                self.output_queues["color"] = encoder.out.createOutputQueue(
                    maxSize=3, blocking=False
                )
            else:
                self.output_queues["color"] = self.cam_rgb.requestOutput(
                    config.color_image_dim,
                    fps=config.fps,
                ).createOutputQueue(maxSize=3, blocking=False)
            print(f"Enabled CAM_A (RGB){' with MJPEG encoding' if config.use_mjpeg else ''}")

            if not config.autofocus:
                ctrl_in = self.cam_rgb.inputControl.createInputQueue()
                ctrl = dai.CameraControl()
                ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
                ctrl.setManualFocus(config.manual_focus)
                ctrl_in.send(ctrl)
                print(f"Autofocus disabled, manual focus set to {config.manual_focus}")

        # Monochrome cameras (CAM_B / CAM_C)
        if config.enable_mono_cameras:
            if dai.CameraBoardSocket.CAM_B in sockets:
                self.cam_mono_left = self.pipeline.create(dai.node.Camera)
                self.cam_mono_left.initialControl.setAutoExposureLimit(32000)   # U36
                cam_socket = dai.CameraBoardSocket.CAM_B
                self.cam_mono_left = self.cam_mono_left.build(cam_socket)

                if config.use_mjpeg:
                    cam_out = self.cam_mono_left.requestOutput(
                        config.monochrome_image_dim,
                        dai.ImgFrame.Type.NV12,
                        fps=config.fps,
                    )
                    encoder = self.pipeline.create(dai.node.VideoEncoder)
                    encoder.setDefaultProfilePreset(
                        config.fps, dai.VideoEncoderProperties.Profile.MJPEG
                    )
                    encoder.setQuality(config.mjpeg_quality)
                    cam_out.link(encoder.input)
                    self.output_queues["mono_left"] = encoder.out.createOutputQueue(
                        maxSize=3, blocking=False
                    )
                else:
                    self.output_queues["mono_left"] = self.cam_mono_left.requestOutput(
                        config.monochrome_image_dim,
                        fps=config.fps,
                    ).createOutputQueue(maxSize=3, blocking=False)
                print("Enabled CAM_B (Monochrome Left)")

            if dai.CameraBoardSocket.CAM_C in sockets:
                self.cam_mono_right = self.pipeline.create(dai.node.Camera)
                self.cam_mono_right.initialControl.setAutoExposureLimit(32000)   # U36
                cam_socket = dai.CameraBoardSocket.CAM_C
                self.cam_mono_right = self.cam_mono_right.build(cam_socket)

                if config.use_mjpeg:
                    cam_out = self.cam_mono_right.requestOutput(
                        config.monochrome_image_dim,
                        dai.ImgFrame.Type.NV12,
                        fps=config.fps,
                    )
                    encoder = self.pipeline.create(dai.node.VideoEncoder)
                    encoder.setDefaultProfilePreset(
                        config.fps, dai.VideoEncoderProperties.Profile.MJPEG
                    )
                    encoder.setQuality(config.mjpeg_quality)
                    cam_out.link(encoder.input)
                    self.output_queues["mono_right"] = encoder.out.createOutputQueue(
                        maxSize=3, blocking=False
                    )
                else:
                    self.output_queues["mono_right"] = self.cam_mono_right.requestOutput(
                        config.monochrome_image_dim,
                        fps=config.fps,
                    ).createOutputQueue(maxSize=3, blocking=False)
                print("Enabled CAM_C (Monochrome Right)")

        # U29: derived streams from the hardware-synced stereo pair. Their queues are NOT part of the
        # all-cameras-present gate in read(): the pair syncs at its own cadence and depth is subsampled.
        # U29b: their encodes run on dedicated threads — the capture loop's 33 ms budget cannot absorb
        # the depth PNG (~23-40 ms measured on PC2); cv2 encodes release the GIL, so the threads run in parallel.
        self._derived_lock = threading.Lock()
        self._derived_latest = {}    # name -> (seq, capture_time, encoded bytes)
        self._derived_emitted = {}   # name -> last seq handed to read()
        self._derived_run = True
        self._derived_threads = []
        self._stitch_feed = None     # U29c: (dev_ts_l, wall_ts, jpeg_l, dev_ts_r, jpeg_r) handed over by read()
        self._stitch_enabled = False
        if (config.enable_stereo_stitch or config.enable_depth) and hasattr(self, "cam_mono_left") and hasattr(self, "cam_mono_right"):
            # U29c: the OAK sits on USB2 (480 Mbps). Raw crossings are the enemy: the U29a GRAY8 pair +
            # Sync (~18 MB/s) plus RAW16 depth @30 (~18 MB/s) saturated the link and starved EVERY stream.
            # Stitch now happens on the host from the two mono MJPEG streams that already cross; the
            # StereoDepth GRAY8 inputs stay device-internal and run at the subsampled depth rate.
            if config.enable_stereo_stitch:
                if config.use_mjpeg:
                    self._stitch_enabled = True
                    print("Enabled stereo stitch (host-side, from the CAM_B/CAM_C MJPEG streams)")
                else:
                    print("[WARN] stereo stitch requires use_mjpeg=True on this build — stitch disabled")
            if config.enable_depth:
                depth_fps = max(1, config.fps // max(1, config.depth_fps_divisor))
                gray_left = self.cam_mono_left.requestOutput(
                    config.monochrome_image_dim, dai.ImgFrame.Type.GRAY8, fps=depth_fps
                )
                gray_right = self.cam_mono_right.requestOutput(
                    config.monochrome_image_dim, dai.ImgFrame.Type.GRAY8, fps=depth_fps
                )
                stereo = self.pipeline.create(dai.node.StereoDepth).build(
                    left=gray_left, right=gray_right, presetMode=dai.node.StereoDepth.PresetMode.DEFAULT
                )
                stereo.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
                self.output_queues["depth"] = stereo.depth.createOutputQueue(maxSize=3, blocking=False)
                print(f"Enabled StereoDepth (uint16 mm, RECTIFIED_LEFT, {depth_fps} fps at the source; GRAY8 inputs device-internal)")

        assert len(self.output_queues) > 0, "No output queues enabled"

        self.pipeline.start()

        if self._stitch_enabled:
            t = threading.Thread(target=self._stitch_worker, name="oak-stitch", daemon=True)
            t.start()
            self._derived_threads.append(t)
        if "depth" in self.output_queues:
            t = threading.Thread(target=self._depth_worker, name="oak-depth", daemon=True)
            t.start()
            self._derived_threads.append(t)

        print(f"[{mount_position}] Pipeline started, waiting for stabilization...")
        time.sleep(2.0)

        for _ in range(10):
            test_frame = None
            for queue_name, q in self.output_queues.items():
                test_frame = q.tryGet()
                if test_frame:
                    print(f"[{mount_position}] First frame received from {queue_name}")
                    break
            if test_frame:
                break
            time.sleep(0.3)
        else:
            print(f"[{mount_position}] Warning: No frames received during init verification")

        if run_as_server:
            self.start_server(port)

    def _stitch_worker(self):
        seq = 0
        last_ts = None
        warned = 0.0
        while self._derived_run:
            feed = self._stitch_feed
            if feed is None:
                time.sleep(0.002)
                continue
            lts, wall, lb, rts, rb = feed
            if lts == last_ts:
                time.sleep(0.002)
                continue
            last_ts = lts
            skew = abs((lts - rts).total_seconds())
            if skew > 0.010:
                now = time.monotonic()
                if now - warned > 10.0:
                    print(f"[WARN] stitch pair skew {skew * 1000:.1f} ms — frame skipped")
                    warned = now
                continue
            try:
                li = cv2.imdecode(np.frombuffer(lb, np.uint8), cv2.IMREAD_GRAYSCALE)
                ri = cv2.imdecode(np.frombuffer(rb, np.uint8), cv2.IMREAD_GRAYSCALE)
                if li is None or ri is None:
                    continue
                ok, buf = cv2.imencode(
                    ".jpg", np.hstack([li, ri]), [int(cv2.IMWRITE_JPEG_QUALITY), self.config.stitch_jpeg_quality]
                )
                if ok:
                    seq += 1
                    with self._derived_lock:
                        self._derived_latest["stereo"] = (seq, wall, buf.tobytes())
            except Exception as e:
                print(f"[ERROR] stitch encode failed: {e}")
                time.sleep(0.1)

    def _depth_worker(self):
        q = self.output_queues["depth"]
        seq = 0
        while self._derived_run:
            df = q.tryGet()
            if df is None:
                time.sleep(0.002)
                continue
            try:
                ok, buf = cv2.imencode(".png", df.getCvFrame(), [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                if ok:
                    cap = time.time() - (dai.Clock.now() - df.getTimestamp()).total_seconds()
                    seq += 1
                    with self._derived_lock:
                        self._derived_latest["depth"] = (seq, cap, buf.tobytes())
            except Exception as e:
                print(f"[ERROR] depth encode failed: {e}")
                time.sleep(0.1)

    def read(self) -> dict[str, Any] | None:
        if not self.pipeline.isRunning():
            print(f"[ERROR] OAK pipeline stopped for {self.mount_position}")
            return None
        if not self.device.isPipelineRunning():
            print(f"[ERROR] OAK device disconnected for {self.mount_position}")
            return None

        # U40 probe (log-only): who loses the frames — device production (devrate), XLink/queue
        # (drainloss/gap), or the read loop (reads/s, none, disc, age_max). One [P40] line per 5 s.
        p = getattr(self, "_p40", None)
        if p is None:
            p = self._p40 = {"t0": time.monotonic(), "reads": 0, "none": {}, "disc": 0,
                             "drainloss": 0, "gap": 0, "seq0": {}, "seq": {}, "age_max": 0.0,
                             "exp_ms": -1.0}
        p["reads"] += 1

        timestamps = {}
        images = {}
        rgb_frame_time = None

        def drain_queue_get_latest(queue, sk=None):
            latest_frame = None
            pulled = 0
            first_seq = last_seq = None
            while True:
                frame = queue.tryGet()
                if frame is None:
                    break
                latest_frame = frame
                pulled += 1
                if sk is not None:
                    try:
                        sq = frame.getSequenceNum()
                        if first_seq is None:
                            first_seq = sq
                        last_seq = sq
                    except Exception:
                        pass
            if sk is not None and pulled:
                try:
                    p["disc"] += pulled - 1
                    if first_seq is not None and last_seq is not None:
                        p["drainloss"] += max(0, (last_seq - first_seq + 1) - pulled)
                        prev = p["seq"].get(sk)
                        if prev is not None and first_seq > prev + 1:
                            p["gap"] += first_seq - prev - 1
                        if sk not in p["seq0"]:
                            p["seq0"][sk] = first_seq
                        p["seq"][sk] = last_seq
                except Exception:
                    pass
            return latest_frame

        def drain_wait_latest(queue, wait_s=0.006, sk=None):
            # U29e: a paced send slot must not die on a drain-empty race — the next frame of a healthy
            # 30 fps queue is at most a few ms away. Wait for it briefly instead of dropping the slot.
            deadline = time.monotonic() + wait_s
            while True:
                frame = drain_queue_get_latest(queue, sk)
                if frame is not None or time.monotonic() >= deadline:
                    if frame is None and sk is not None:
                        p["none"][sk] = p["none"].get(sk, 0) + 1
                    return frame
                time.sleep(0.001)

        expected_cameras = set(self.output_queues.keys()) - {"stereo_pair", "depth"}  # U29: derived streams never block
        received_cameras = set()

        if "color" in self.output_queues:
            try:
                rgb_frame = drain_wait_latest(self.output_queues["color"], sk="c")
                if rgb_frame is None:
                    return None
                rgb_frame_time = rgb_frame.getTimestamp()
                read_time = time.time()
                frame_age = (dai.Clock.now() - rgb_frame_time).total_seconds()
                capture_time = read_time - frame_age
                try:  # U40 probe: last color exposure, ms
                    p["exp_ms"] = rgb_frame.getExposureTime().total_seconds() * 1000.0
                except Exception:
                    pass

                if self._use_mjpeg:
                    images[self.mount_position] = bytes(rgb_frame.getData())
                else:
                    images[self.mount_position] = rgb_frame.getCvFrame()[..., ::-1]
                timestamps[self.mount_position] = capture_time
                received_cameras.add("color")
            except Exception as e:
                print(f"[ERROR] Failed to read color frame from {self.mount_position}: {e}")
                return None

        if "mono_left" in self.output_queues:
            try:
                mono_left_frame = drain_wait_latest(self.output_queues["mono_left"], sk="l")
                if mono_left_frame is None:
                    return None
                mono_left_frame_time = mono_left_frame.getTimestamp()
                read_time = time.time()
                frame_age = (dai.Clock.now() - mono_left_frame_time).total_seconds()
                capture_time = read_time - frame_age

                key = f"{self.mount_position}_left_mono"
                if self._use_mjpeg:
                    images[key] = bytes(mono_left_frame.getData())
                else:
                    images[key] = mono_left_frame.getCvFrame()
                timestamps[key] = capture_time
                received_cameras.add("mono_left")
            except Exception as e:
                print(f"[ERROR] Failed to read mono_left frame from {self.mount_position}: {e}")
                return None

        if "mono_right" in self.output_queues:
            try:
                mono_right_frame = drain_wait_latest(self.output_queues["mono_right"], sk="r")
                if mono_right_frame is None:
                    return None
                mono_right_frame_time = mono_right_frame.getTimestamp()
                read_time = time.time()
                frame_age = (dai.Clock.now() - mono_right_frame_time).total_seconds()
                capture_time = read_time - frame_age

                key = f"{self.mount_position}_right_mono"
                if self._use_mjpeg:
                    images[key] = bytes(mono_right_frame.getData())
                else:
                    images[key] = mono_right_frame.getCvFrame()
                timestamps[key] = capture_time
                received_cameras.add("mono_right")
            except Exception as e:
                print(f"[ERROR] Failed to read mono_right frame from {self.mount_position}: {e}")
                return None

        if self._stitch_enabled:  # U29c: hand the hw-synced mono pair to the stitch thread — no extra USB traffic
            lk, rk = f"{self.mount_position}_left_mono", f"{self.mount_position}_right_mono"
            if lk in images and rk in images:
                self._stitch_feed = (mono_left_frame_time, timestamps[lk], images[lk], mono_right_frame_time, images[rk])

        if self._derived_threads:  # U29b: pick up the encoder threads' latest results, each frame once
            with self._derived_lock:
                snap = dict(self._derived_latest)
            for name, (seq, cap, payload) in snap.items():
                if self._derived_emitted.get(name) == seq:
                    continue
                self._derived_emitted[name] = seq
                images[f"{self.mount_position}_{name}"] = payload
                timestamps[f"{self.mount_position}_{name}"] = cap

        if received_cameras != expected_cameras:
            missing = expected_cameras - received_cameras
            print(f"[ERROR] Missing frames from cameras: {missing} for {self.mount_position}")
            return None

        if rgb_frame_time is not None:
            frame_age = (dai.Clock.now() - rgb_frame_time).total_seconds()
            if frame_age > p["age_max"]:  # U40 probe
                p["age_max"] = frame_age
            if frame_age > 0.1:
                print(
                    f"[{self.mount_position}] OAK frame age too large: {frame_age * 1000:.1f}ms"
                )

        try:  # U40 probe: 5 s window report, then reset
            _win = time.monotonic() - p["t0"]
            if _win >= 5.0:
                _rates = []
                for _k in ("c", "l", "r"):
                    _s0 = p["seq0"].get(_k); _s1 = p["seq"].get(_k)
                    _rates.append("%.1f" % ((_s1 - _s0) / _win) if _s0 is not None and _s1 is not None and _s1 > _s0 else "?")
                try:
                    _temp = "%.1fC" % self.device.getChipTemperature().average
                except Exception:
                    _temp = "?"
                try:
                    _usb = str(self.device.getUsbSpeed()).split(".")[-1]
                except Exception:
                    _usb = "?"
                _n = p["none"]
                print("[P40] reads/s=%.1f none=c%d/l%d/r%d disc=%d drainloss=%d gap=%d devrate=%s exp=%.1fms temp=%s usb=%s age_max=%.0fms"
                      % (p["reads"] / _win, _n.get("c", 0), _n.get("l", 0), _n.get("r", 0),
                         p["disc"], p["drainloss"], p["gap"], "/".join(_rates),
                         p["exp_ms"], _temp, _usb, p["age_max"] * 1000))
                self._p40 = None
        except Exception as _e:
            self._p40 = None
            if not getattr(self, "_p40_warned", False):
                self._p40_warned = True
                print(f"[P40] report failed ({_e}); probe continues")

        return {"timestamps": timestamps, "images": images}

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        spaces = {}
        if self.config.enable_color:
            spaces["color_image"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.config.color_image_dim[1], self.config.color_image_dim[0], 3),
                dtype=np.uint8,
            )
        if self.config.enable_mono_cameras:
            spaces["mono_left_image"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.config.monochrome_image_dim[1], self.config.monochrome_image_dim[0]),
                dtype=np.uint8,
            )
            spaces["mono_right_image"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.config.monochrome_image_dim[1], self.config.monochrome_image_dim[0]),
                dtype=np.uint8,
            )
        return gym.spaces.Dict(spaces)

    def close(self):
        self._derived_run = False
        if self._run_as_server:
            self.stop_server()
        if hasattr(self, "pipeline") and self.pipeline.isRunning():
            self.pipeline.stop()
        self.device.close()

    def run_server(self):
        if not self._run_as_server:
            raise ValueError("run_as_server must be True to call run_server()")
        while True:
            frame = self.read()
            if frame is None:
                continue
            msg = self.serialize(frame)
            self.send_message({self.mount_position: msg})

    def __del__(self):
        self.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true", help="Run as server")
    parser.add_argument("--host", type=str, default="localhost", help="Server IP address")
    parser.add_argument("--port", type=int, default=5555, help="Port number")
    parser.add_argument("--device-id", type=str, default=None, help="Specific device ID")
    parser.add_argument(
        "--enable-mono", action="store_true", help="Enable monochrome cameras (CAM_B & CAM_C)"
    )
    parser.add_argument("--mount-position", type=str, default="ego_view", help="Mount position")
    parser.add_argument("--show-image", action="store_true", help="Display images")
    parser.add_argument("--use-mjpeg", action="store_true", help="Use MJPEG encoding on-device")
    parser.add_argument(
        "--mjpeg-quality", type=int, default=80, help="MJPEG quality 1-100 (default: 80)"
    )
    args = parser.parse_args()

    oak_config = OAKConfig()
    if args.enable_mono:
        oak_config.enable_mono_cameras = True
    if args.use_mjpeg:
        oak_config.use_mjpeg = True
        oak_config.mjpeg_quality = args.mjpeg_quality

    if args.server:
        oak = OAKSensor(
            run_as_server=True,
            port=args.port,
            config=oak_config,
            device_id=args.device_id,
            mount_position=args.mount_position,
        )
        print(f"Starting OAK server on port {args.port}")
        oak.run_server()
    else:
        oak = OAKSensor(run_as_server=False, config=oak_config, device_id=args.device_id)
        print("Running OAK camera in standalone mode")

        while True:
            frame = oak.read()
            if frame is None:
                print("Waiting for frame...")
                time.sleep(0.5)
                continue

            if args.show_image:
                for key, img in frame.get("images", {}).items():
                    if isinstance(img, np.ndarray):
                        cv2.imshow(key, img[..., ::-1] if img.ndim == 3 and img.shape[2] == 3 else img)
                if cv2.waitKey(1) == ord("q"):
                    break

            time.sleep(0.01)

        cv2.destroyAllWindows()
        oak.close()
