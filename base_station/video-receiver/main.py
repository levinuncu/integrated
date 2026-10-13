import argparse
import signal
import os

from pathlib import Path
from threading import Event
from time import monotonic

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from rolling_recording import RollingRecording


class VideoReceiver:
    def __init__(
        self,
        port: int = 5004,
        relay_host: str = "127.0.0.1",
        relay_port: int = 5006,
        recording: RollingRecording | None = None,
    ) -> None:
        Gst.init(None)
        self._port = port
        self._relay_host = relay_host
        self._relay_port = relay_port
        self._recording = recording or RollingRecording(Path("/recordings"))
        self._stop_requested = Event()
        self._pipeline = Gst.Pipeline.new("base-video-receiver")
        self._build_pipeline()

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._request_stop)
        signal.signal(signal.SIGTERM, self._request_stop)
        self._recording.cleanup()

        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Cannot start the RTP receiver")

        bus = self._pipeline.get_bus()
        last_cleanup = monotonic()
        try:
            while not self._stop_requested.is_set():
                message = bus.timed_pop_filtered(
                    Gst.SECOND,
                    Gst.MessageType.ERROR | Gst.MessageType.EOS,
                )
                if message is not None:
                    if message.type == Gst.MessageType.ERROR:
                        error, details = message.parse_error()
                        raise RuntimeError(f"GStreamer: {error.message} ({details})")
                    if message.type == Gst.MessageType.EOS:
                        break

                if monotonic() - last_cleanup >= 10:
                    self._recording.cleanup()
                    last_cleanup = monotonic()
        finally:
            self._pipeline.send_event(Gst.Event.new_eos())
            bus.timed_pop_filtered(
                5 * Gst.SECOND,
                Gst.MessageType.ERROR | Gst.MessageType.EOS,
            )
            self._pipeline.set_state(Gst.State.NULL)
            self._recording.cleanup()

    def _request_stop(self, *_args) -> None:
        self._stop_requested.set()

    def _build_pipeline(self) -> None:
        source = self._add("udpsrc", "drone-rtp-input")
        source.set_property("port", self._port)
        source.set_property("reuse", False)
        source.set_property(
            "caps",
            Gst.Caps.from_string(
                "application/x-rtp,media=(string)video,clock-rate=(int)90000,"
                "encoding-name=(string)H264,payload=(int)96"
            ),
        )

        jitter = self._add("rtpjitterbuffer", "wifi-jitter-buffer")
        jitter.set_property("latency", 100)

        depayloader = self._add("rtph264depay", "rtp-depacketizer")
        parser = self._add("h264parse", "video-parser")
        splitter = self._add("tee", "encoded-video-splitter")

        self._link(source, jitter, depayloader, parser, splitter)
        self._create_recording_branch(splitter)
        self._create_webrtc_relay_branch(splitter)

    def _create_recording_branch(self, splitter) -> None:
        queue = self._add("queue", "recording-queue")
        parser = self._add("h264parse", "mp4-parser")
        recorder = self._add("splitmuxsink", "ring-mp4-recorder")
        self._recording.configure(recorder, Gst.SECOND)

        self._link(queue, parser, recorder)
        self._connect_tee(splitter, queue)

    def _create_webrtc_relay_branch(self, splitter) -> None:
        queue = self._add("queue", "webrtc-relay-queue")
        queue.set_property("leaky", 2)
        queue.set_property("max-size-buffers", 8)
        queue.set_property("max-size-bytes", 0)
        queue.set_property("max-size-time", 0)

        parser = self._add("h264parse", "relay-h264-parser")
        payloader = self._add("rtph264pay", "mediamtx-rtp-payloader")
        payloader.set_property("pt", 96)
        payloader.set_property("config-interval", -1)
        payloader.set_property("mtu", 1200)

        sink = self._add("udpsink", "mediamtx-udp-relay")
        sink.set_property("host", self._relay_host)
        sink.set_property("port", self._relay_port)
        sink.set_property("sync", False)
        sink.set_property("async", False)

        self._link(queue, parser, payloader, sink)
        self._connect_tee(splitter, queue)

    def _add(self, factory: str, name: str):
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"Missing GStreamer plugin: {factory}")
        self._pipeline.add(element)
        return element

    @staticmethod
    def _link(*elements) -> None:
        for src, dst in zip(elements, elements[1:]):
            if not src.link(dst):
                raise RuntimeError(f"Cannot link {src.get_name()} -> {dst.get_name()}")

    @staticmethod
    def _connect_tee(splitter, queue) -> None:
        pad = splitter.request_pad_simple("src_%u")
        sink_pad = queue.get_static_pad("sink")
        if pad is None or sink_pad is None:
            raise RuntimeError("Could not acquire tee pads")
        if pad.link(sink_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Could not link tee branch")


def main() -> None:
    VideoReceiver(
        port=int(os.getenv("PORT", "5004")),
        relay_host=os.getenv("VIDEO_RELAY_HOST", "127.0.0.1"),
        relay_port=int(os.getenv("VIDEO_RELAY_PORT", "5006")),
        recording=RollingRecording(
            directory=Path("/recordings"),
            name="rgb",
            segment_seconds=int(os.getenv("RECORD_SEGMENT_SECONDS", "10")),
            keep_seconds=int(os.getenv("RECORD_KEEP_SECONDS", "600")),
        ),
    ).run()


if __name__ == "__main__":
    main()
