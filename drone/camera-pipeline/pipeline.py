from collections import OrderedDict
from itertools import count
from pathlib import Path
from signal import SIGINT, SIGTERM, signal
from threading import Event, Lock, Thread
from time import monotonic, sleep

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from camera import Camera, CameraFrame
from frame_bundle import FrameBundle
from frame_publisher import FramePublisher
from frame_synchronizer import FrameSynchronizer
from stream_destination import StreamDestination
from rolling_recording import RollingRecording


class Pipeline:
    def __init__(
        self,
        rgb_camera: Camera,
        thermal_camera: Camera,
        rgb_frame_rate: int = 10,
        thermal_frame_rate: int = 8,
        ai_frame_rate: int = 2,
        recording: RollingRecording | None = None,
        stream_destination: StreamDestination | None = None,
        frame_publisher: FramePublisher | None = None,
    ) -> None:
        if min(rgb_frame_rate, thermal_frame_rate, ai_frame_rate) <= 0:
            raise ValueError("Frame rates must be positive")
        if ai_frame_rate > rgb_frame_rate:
            raise ValueError("AI frame rate cannot exceed RGB frame rate")

        Gst.init(None)
        self._rgb_camera = rgb_camera
        self._thermal_camera = thermal_camera
        self._rgb_frame_rate = rgb_frame_rate
        self._thermal_frame_rate = thermal_frame_rate
        self._ai_interval_ns = Gst.SECOND // ai_frame_rate
        self._frame_duration_ns = Gst.SECOND // rgb_frame_rate
        self._recording = recording or RollingRecording(Path("/recordings"))
        self._recording.cleanup()
        self._stream_destination = stream_destination
        self._frame_publisher = frame_publisher

        self._synchronizer = FrameSynchronizer(max_time_difference_ms=100)

        self._pipeline: Gst.Pipeline | None = None
        self._source: Gst.Element | None = None
        self._first_rgb_timestamp_ns: int | None = None
        self._last_ai_timestamp_ns: int | None = None
        self._ai_frame_count = 0
        self._callback_error: Exception | None = None

        self._rgb_metadata: OrderedDict[int, tuple[int, int]] = OrderedDict()
        self._metadata_lock = Lock()

        self._stop_thermal = Event()
        self._thermal_ready = Event()
        self._thermal_thread: Thread | None = None
        self._thermal_error: Exception | None = None
        self._stop_requested = Event()

    def run(self, number_of_frames: int | None = None) -> None:
        if number_of_frames is not None and number_of_frames <= 0:
            raise ValueError("number_of_frames must be positive")

        signal(SIGINT, lambda *_: self._stop_requested.set())
        signal(SIGTERM, lambda *_: self._stop_requested.set())

        self._create_pipeline()
        assert self._pipeline is not None
        assert self._source is not None

        try:
            if self._frame_publisher is not None:
                self._frame_publisher.start()

            result = self._pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Cannot start GStreamer pipeline")

            self._start_thermal_reader()

            started_at = monotonic()
            frame_numbers = count() if number_of_frames is None else range(number_of_frames)
            for frame_number in frame_numbers:
                if self._stop_requested.is_set():
                    break
                next_capture = started_at + frame_number / self._rgb_frame_rate
                if self._stop_requested.wait(max(0, next_capture - monotonic())):
                    break

                if frame_number % (self._rgb_frame_rate * 10) == 0:
                    self._recording.cleanup()

                self._raise_thermal_error()
                self._raise_callback_error()
                self._push_rgb_frame(self._rgb_camera.capture())

            self._check_flow(self._source.emit("end-of-stream"))
            self._wait_for_eos()
            self._raise_callback_error()
        finally:
            self._stop_thermal_reader()
            self._pipeline.set_state(Gst.State.NULL)
            if self._frame_publisher is not None:
                self._frame_publisher.close()

            self._rgb_camera.close()
            self._thermal_camera.close()

    def _create_pipeline(self) -> None:
        self._pipeline = Gst.Pipeline.new("camera-pipeline")
        self._source = self._add("appsrc", "rgb-camera-input")

        self._source.set_property("caps", Gst.Caps.from_string(
            "video/x-raw,format=RGB,"
            f"width={self._rgb_camera.width},height={self._rgb_camera.height},"
            f"framerate={self._rgb_frame_rate}/1"
        ))
        self._source.set_property("format", Gst.Format.TIME)
        self._source.set_property("is-live", True)
        self._source.set_property("block", True)
        self._source.set_property(
            "max-bytes", self._rgb_camera.width * self._rgb_camera.height * 3 * 2
        )

        splitter = self._add("tee", "rgb-splitter")
        self._link(self._source, splitter)

        self._create_video_branch(splitter)
        self._create_ai_branch(splitter)

    def _create_video_branch(self, rgb_splitter: Gst.Element) -> None:
        queue = self._add("queue", "video-input-queue")
        converter = self._add("videoconvert", "video-color-converter")
        encoder = self._add("x264enc", "h264-encoder")
        browser_caps = self._add("capsfilter", "browser-compatible-h264")
        browser_caps.set_property(
            "caps", Gst.Caps.from_string("video/x-h264,profile=baseline")
        )
        parser = self._add("h264parse", "h264-parser")

        h264_splitter = self._add("tee", "h264-splitter")

        encoder.set_property("bitrate", 2000)
        encoder.set_property("speed-preset", 1)
        encoder.set_property("tune", 0x04)
        encoder.set_property("key-int-max", self._rgb_frame_rate)

        self._link(queue, converter, encoder, browser_caps, parser, h264_splitter)
        self._connect_tee(rgb_splitter, queue)

        self._create_recording_branch(h264_splitter)
        if self._stream_destination is not None:
            self._create_streaming_branch(h264_splitter, self._stream_destination)

    def _create_recording_branch(self, h264_splitter: Gst.Element) -> None:
        queue = self._add("queue", "recording-queue")
        parser = self._add("h264parse", "recording-h264-parser")
        recorder = self._add("splitmuxsink", "mp4-ring-recorder")

        self._recording.configure(recorder, Gst.SECOND)

        self._link(queue, parser, recorder)
        self._connect_tee(h264_splitter, queue)

    def _create_streaming_branch(
        self, h264_splitter: Gst.Element, destination: StreamDestination
    ) -> None:
        queue = self._add("queue", "streaming-queue")
        queue.set_property("leaky", 2)
        queue.set_property("max-size-buffers", 4)
        queue.set_property("max-size-bytes", 0)
        queue.set_property("max-size-time", 0)

        payloader = self._add("rtph264pay", "h264-rtp-packetizer")
        payloader.set_property("pt", 96)
        payloader.set_property("config-interval", -1)
        payloader.set_property("mtu", 1200)

        network_sink = self._add("udpsink", "rtp-udp-output")
        network_sink.set_property("host", destination.host)
        network_sink.set_property("port", destination.port)
        network_sink.set_property("sync", False)
        network_sink.set_property("async", False)

        self._link(queue, payloader, network_sink)
        self._connect_tee(h264_splitter, queue)

    def _create_ai_branch(self, splitter: Gst.Element) -> None:
        queue = self._add("queue", "ai-queue")
        queue.set_property("max-size-buffers", 1)
        queue.set_property("max-size-bytes", 0)
        queue.set_property("max-size-time", 0)
        queue.set_property("leaky", 2)

        scaler = self._add("videoscale", "ai-scaler")
        converter = self._add("videoconvert", "ai-color-converter")
        caps_filter = self._add("capsfilter", "ai-output-format")
        caps_filter.set_property("caps", Gst.Caps.from_string(
            "video/x-raw,format=RGB,width=320,height=180"
        ))

        sink = self._add("appsink", "ai-frame-output")
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 1)
        sink.set_property("drop", True)
        sink.set_property("wait-on-eos", False)
        sink.connect("new-sample", self._on_ai_frame)

        self._link(queue, scaler, converter, caps_filter, sink)
        self._connect_tee(splitter, queue)

    def _push_rgb_frame(self, frame: CameraFrame) -> None:
        if not isinstance(frame.data, bytes):
            raise TypeError("RGB camera must produce bytes")
        if len(frame.data) != frame.width * frame.height * 3:
            raise ValueError("RGB data must contain exactly 3 bytes per pixel")

        if self._first_rgb_timestamp_ns is None:
            self._first_rgb_timestamp_ns = frame.timestamp_ns

        pts = frame.timestamp_ns - self._first_rgb_timestamp_ns
        buffer = Gst.Buffer.new_allocate(None, len(frame.data), None)
        buffer.fill(0, frame.data)
        buffer.pts = pts
        buffer.duration = self._frame_duration_ns

        with self._metadata_lock:
            self._rgb_metadata[pts] = (frame.frame_id, frame.timestamp_ns)
            if len(self._rgb_metadata) > 128:
                self._rgb_metadata.popitem(last=False)

        assert self._source is not None
        self._check_flow(self._source.emit("push-buffer", buffer))

    def _on_ai_frame(self, sink: Gst.Element) -> Gst.FlowReturn:
        try:
            sample = sink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.EOS

            buffer = sample.get_buffer()
            if buffer is None:
                return Gst.FlowReturn.ERROR

            with self._metadata_lock:
                metadata = self._rgb_metadata.pop(buffer.pts, None)

            if metadata is None:
                return Gst.FlowReturn.OK

            frame_id, timestamp_ns = metadata
            if (
                self._last_ai_timestamp_ns is not None
                and timestamp_ns - self._last_ai_timestamp_ns < self._ai_interval_ns
            ):
                return Gst.FlowReturn.OK
            self._last_ai_timestamp_ns = timestamp_ns

            caps = sample.get_caps().get_structure(0)
            width = caps.get_value("width")
            height = caps.get_value("height")

            rgb_frame = CameraFrame(
                camera_id="rgb",
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                width=width,
                height=height,
                data=buffer.extract_dup(0, buffer.get_size()),
            )

            bundle = self._synchronizer.pair_with_rgb(rgb_frame)
            self._handle_ai_bundle(bundle)
            return Gst.FlowReturn.OK
        except Exception as error:
            self._callback_error = error
            return Gst.FlowReturn.ERROR

    def _handle_ai_bundle(self, bundle: FrameBundle) -> None:
        self._ai_frame_count += 1
        if self._frame_publisher is not None:
            self._frame_publisher.publish(bundle)

        rgb = bundle.rgb_frame
        thermal = bundle.thermal_frame
        if thermal is None:
            return

        delta_ms = bundle.time_difference_ns / 1_000_000

    def _start_thermal_reader(self) -> None:
        self._thermal_thread = Thread(
            target=self._read_thermal_frames,
            name="thermal-reader",
            daemon=True,
        )
        self._thermal_thread.start()
        if not self._thermal_ready.wait(timeout=2):
            raise TimeoutError("No thermal measurement received within 2 seconds")
        self._raise_thermal_error()

    def _read_thermal_frames(self) -> None:
        try:
            started_at = monotonic()
            frame_number = 0
            while not self._stop_thermal.is_set():
                target_time = started_at + frame_number / self._thermal_frame_rate
                if self._stop_thermal.wait(max(0, target_time - monotonic())):
                    break

                thermal_frame = self._thermal_camera.capture()
                self._synchronizer.add_thermal_frame(thermal_frame)
                self._thermal_ready.set()
                frame_number += 1
        except Exception as error:
            self._thermal_error = error
            self._thermal_ready.set()

    def _stop_thermal_reader(self) -> None:
        self._stop_thermal.set()
        if self._thermal_thread is not None:
            self._thermal_thread.join(timeout=2)

    def _wait_for_eos(self) -> None:
        assert self._pipeline is not None
        message = self._pipeline.get_bus().timed_pop_filtered(
            30 * Gst.SECOND,
            Gst.MessageType.ERROR | Gst.MessageType.EOS,
        )
        if message is None:
            raise TimeoutError("GStreamer did not finish in 30 seconds")
        if message.type == Gst.MessageType.ERROR:
            error, details = message.parse_error()
            raise RuntimeError(f"GStreamer error: {error.message} ({details})")

    def _raise_thermal_error(self) -> None:
        if self._thermal_error is not None:
            raise RuntimeError("Thermal camera reader failed") from self._thermal_error

    def _raise_callback_error(self) -> None:
        if self._callback_error is not None:
            raise RuntimeError("Failed to process an AI frame") from self._callback_error

    def _add(self, factory: str, name: str) -> Gst.Element:
        assert self._pipeline is not None
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"Missing GStreamer plugin: {factory}")
        self._pipeline.add(element)
        return element

    @staticmethod
    def _link(*elements: Gst.Element) -> None:
        for source, destination in zip(elements, elements[1:]):
            if not source.link(destination):
                raise RuntimeError(
                    f"Cannot link {source.get_name()} -> {destination.get_name()}"
                )

    @staticmethod
    def _connect_tee(splitter: Gst.Element, queue: Gst.Element) -> None:
        source_pad = splitter.request_pad_simple("src_%u")
        destination_pad = queue.get_static_pad("sink")
        if source_pad is None or destination_pad is None:
            raise RuntimeError("Cannot obtain tee pads")
        if source_pad.link(destination_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Cannot connect tee branch")

    @staticmethod
    def _check_flow(result: Gst.FlowReturn) -> None:
        if result != Gst.FlowReturn.OK:
            raise RuntimeError(f"GStreamer flow failed: {result.value_nick}")
