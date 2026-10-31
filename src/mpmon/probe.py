"""IoT node for the in-line flow probe: sensors, edge analysis, MQTT telemetry, alarms, wiper.

One measurement window (e.g. 10 s of video) produces one MQTT message::

    mpmon/<station>/probe/telemetry   counts, concentration with 95 % interval, health
    mpmon/<station>/probe/alarm       only when something is wrong
    mpmon/<station>/probe/cmd         downlink: {"cmd": "wipe"}, {"cmd": "set_flow", "ml_min": 0.5},
                                      {"cmd": "calibrate", "recovery": 0.62}

The node runs its own rules at the edge, so it does not depend on the cloud being reachable:
low flow (pump stopped or tube clogged) skips analysis; a dim window or many stuck objects
triggers the wiper; many bubbles raise a degassing warning; a concentration alarm is raised only
when the *lower* 95 % bound exceeds the action level, so Poisson noise in a short window does not
cause false alarms.

``*Sim`` classes are used in the simulation. ``FlowSensorSLF3S``, ``WiperServo`` and
``MqttPublisher`` are the Raspberry Pi versions; they are written to the data sheets but have not
been tested on hardware.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .flow import Probe, Water, Window, classify, poisson_ci, run_window, stuck_particles, summarise
from .ml import Model

SCHEMA = """
CREATE TABLE IF NOT EXISTS probe_windows (station TEXT, seq INTEGER, t REAL, flow_ml_min REAL, volume_ml REAL,
    plastic INTEGER, per_l REAL, per_l_lo REAL, per_l_hi REAL, transmission REAL, stuck INTEGER,
    bubbles INTEGER, alarms TEXT, payload TEXT, PRIMARY KEY (station, seq));
"""


# ---------------------------------------------------------------- transport

class Broker:
    """In-memory stand-in for an MQTT broker (topic -> subscribers), keeps every message."""

    def __init__(self):
        self.messages: list[tuple[str, dict]] = []
        self.subs: dict[str, list[Callable[[dict], None]]] = {}

    def publish(self, topic: str, payload: dict) -> None:
        self.messages.append((topic, payload))
        for cb in self.subs.get(topic, []):
            cb(payload)

    def subscribe(self, topic: str, cb: Callable[[dict], None]) -> None:
        self.subs.setdefault(topic, []).append(cb)


class MqttPublisher:  # pragma: no cover - needs paho-mqtt and a broker
    """Same interface as ``Broker`` on a real MQTT broker (``pip install paho-mqtt``)."""

    def __init__(self, host: str = "localhost", port: int = 1883):
        import paho.mqtt.client as mqtt
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.connect(host, port)
        self.client.loop_start()
        self.subs: dict[str, list[Callable[[dict], None]]] = {}
        self.client.on_message = lambda c, u, m: [cb(json.loads(m.payload)) for cb in self.subs.get(m.topic, [])]

    def publish(self, topic: str, payload: dict) -> None:
        self.client.publish(topic, json.dumps(payload), qos=1)

    def subscribe(self, topic: str, cb: Callable[[dict], None]) -> None:
        self.subs.setdefault(topic, []).append(cb)
        self.client.subscribe(topic, qos=1)


# ---------------------------------------------------------------- sensors and actuators

@dataclass
class FlowSensorSim:
    setpoint_ml_min: float
    noise: float = 0.02
    fault: bool = False  # pump stopped / tube clogged

    def read(self, rng: np.random.Generator) -> float:
        return 0.0 if self.fault else float(self.setpoint_ml_min * (1 + rng.normal(0, self.noise)))


class FlowSensorSLF3S:  # pragma: no cover - hardware
    """Sensirion SLF3S-1300F liquid flow sensor on I2C (address 0x08), water calibration."""

    def __init__(self, bus: int = 1):
        from smbus2 import SMBus, i2c_msg
        self.bus, self.msg = SMBus(bus), i2c_msg
        self.bus.i2c_rdwr(self.msg.write(0x08, [0x36, 0x08]))  # start continuous measurement, water
        time.sleep(0.05)

    def read(self, rng=None) -> float:
        r = self.msg.read(0x08, 9)
        self.bus.i2c_rdwr(r)
        b = list(r)
        raw = int.from_bytes(bytes(b[0:2]), "big", signed=True)
        return raw / 500.0  # ml/min, scale factor 500 for the -1300F


@dataclass
class WiperSim:
    on_wipe: Callable[[], None]
    count: int = 0

    def wipe(self) -> None:
        self.count += 1
        self.on_wipe()


class WiperServo:  # pragma: no cover - hardware
    """Silicone wiper blade on a micro servo, swept across the window and back."""

    def __init__(self, gpio: int = 18):
        from gpiozero import AngularServo
        self.servo = AngularServo(gpio, min_angle=0, max_angle=180)
        self.count = 0

    def wipe(self) -> None:
        for a in (170, 10, 170, 10):
            self.servo.angle = a
            time.sleep(0.4)
        self.servo.detach()
        self.count += 1


# ---------------------------------------------------------------- the node

@dataclass
class NodeConfig:
    action_level_per_l: float = 50.0  # site-defined; microplastics have no regulatory limit
    recovery: float = 1.0  # from a spike calibration (flow.spike_recovery)
    min_flow_fraction: float = 0.5
    wipe_below_transmission: float = 0.7
    wipe_above_stuck: int = 6
    bubble_warn_per_ml: float = 100.0
    concentration_factor: float = 1.0  # pre-concentration upstream of the probe (e.g. 1000x)


@dataclass
class ProbeNode:
    station: str
    probe: Probe
    model: Model
    transport: Broker
    flow_sensor: FlowSensorSim
    wiper: WiperSim
    config: NodeConfig = field(default_factory=NodeConfig)
    ref_background: float | None = None
    db: sqlite3.Connection = field(default_factory=lambda: sqlite3.connect(":memory:"))
    seq: int = 0

    def __post_init__(self):
        self.db.executescript(SCHEMA)
        self.transport.subscribe(self.topic("cmd"), self.handle_command)

    def topic(self, kind: str) -> str:
        return f"mpmon/{self.station}/probe/{kind}"

    def handle_command(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        if cmd == "wipe":
            self.wiper.wipe()
        elif cmd == "set_flow":
            self.flow_sensor.setpoint_ml_min = float(msg["ml_min"])
        elif cmd == "calibrate":
            self.config.recovery = float(msg["recovery"])

    def measure(self, dets, n_frames: int, backgrounds: list[float],
                flow_ml_min: float) -> dict:
        """Turn one window's detections into a payload; publish it and any alarms; store it."""
        self.seq += 1
        cfg = self.config
        alarms: list[str] = []
        payload: dict = {"station": self.station, "seq": self.seq, "flow_ml_min": round(flow_ml_min, 3)}
        if flow_ml_min < cfg.min_flow_fraction * self.flow_sensor.setpoint_ml_min:
            alarms.append("flow_low")
            payload.update(status="no_flow", alarms=alarms)
            self._store(payload, 0, float("nan"), float("nan"), float("nan"), float("nan"), 0, 0, 0.0)
            self.transport.publish(self.topic("telemetry"), payload)
            self.transport.publish(self.topic("alarm"), {"seq": self.seq, "alarms": alarms})
            return payload
        s = summarise(dets, self.probe, n_frames, backgrounds, self.ref_background)
        volume_ml = flow_ml_min * s["duration_s"] / 60
        eff_l = volume_ml * cfg.recovery * cfg.concentration_factor / 1000
        lo, hi = poisson_ci(s["plastic"])
        per_l, per_l_lo, per_l_hi = s["plastic"] / eff_l, lo / eff_l, hi / eff_l
        if per_l_lo > cfg.action_level_per_l:
            alarms.append("concentration_high")
        if s["transmission"] < cfg.wipe_below_transmission or s["stuck_objects"] > cfg.wipe_above_stuck:
            alarms.append("window_fouled")
        if s["bubbles"] >= 3 and s["bubbles"] / max(volume_ml, 1e-9) > cfg.bubble_warn_per_ml:
            alarms.append("bubbles_check_degasser")
        payload.update(
            status="ok", duration_s=s["duration_s"], volume_ml=round(volume_ml, 4),
            plastic=s["plastic"], plastic_by_class={k: v for k, v in s["by_class"].items() if k in ("PE", "PP", "PS", "PET")},
            plastic_by_size_um=s["plastic_by_size_um"],
            per_l=round(per_l, 1), per_l_ci95=[round(per_l_lo, 1), round(per_l_hi, 1)],
            health={"transmission": round(s["transmission"], 3), "stuck_objects": s["stuck_objects"],
                    "bubbles": s["bubbles"], "recovery_used": cfg.recovery},
            alarms=alarms,
        )
        self._store(payload, s["plastic"], per_l, per_l_lo, per_l_hi, s["transmission"], s["stuck_objects"],
                    s["bubbles"], volume_ml)
        self.transport.publish(self.topic("telemetry"), payload)
        if alarms:
            self.transport.publish(self.topic("alarm"), {"seq": self.seq, "alarms": alarms})
        if "window_fouled" in alarms:
            self.wiper.wipe()  # local rule: do not wait for the cloud
        return payload

    def _store(self, p, plastic, per_l, lo, hi, transm, stuck, bubbles, volume_ml) -> None:
        self.db.execute("INSERT OR REPLACE INTO probe_windows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (self.station, self.seq, time.time(), p["flow_ml_min"], volume_ml, plastic, per_l, lo, hi,
                         transm, stuck, bubbles, ",".join(p.get("alarms", [])), json.dumps(p)))
        self.db.commit()


# ---------------------------------------------------------------- simulated deployment

@dataclass
class Scenario:
    """What happens to the water and the hardware in each window of the demo."""
    water: Water
    fouling_per_window: float = 0.9  # transmission multiplier per window
    stuck_per_window: int = 2
    events: dict[int, str] = field(default_factory=dict)  # window -> "spike" | "bubbles" | "pump_fault"


def simulate_deployment(node: ProbeNode, scenario: Scenario, n_windows: int, n_frames: int,
                        rng: np.random.Generator) -> list[dict]:
    """Run the node against a simulated probe whose window fouls and gets wiped by the node itself."""
    state = {"T": 1.0, "stuck": []}

    def on_wipe():
        state["T"], state["stuck"] = 0.98, []

    node.wiper.on_wipe = on_wipe
    if node.ref_background is None:  # commissioning: record the clean-window background
        clean = run_window(node.probe, Water(0, 0), 5, rng)
        node.ref_background = float(np.median(clean.backgrounds))
    out = []
    for w in range(n_windows):
        ev = scenario.events.get(w, "")
        water = scenario.water
        if ev == "spike":
            water = Water(water.plastic_per_ml * 4, water.natural_per_ml, water.bubbles_per_ml,
                          water.size_range_um, water.turbidity, water.weathering)
        elif ev == "bubbles":
            water = Water(water.plastic_per_ml, water.natural_per_ml, 2000, water.size_range_um,
                          water.turbidity, water.weathering)
        node.flow_sensor.fault = ev == "pump_fault"
        flow = node.flow_sensor.read(rng)
        if flow < node.config.min_flow_fraction * node.flow_sensor.setpoint_ml_min:
            payload = node.measure([], n_frames, [], flow)
        else:
            run = run_window(node.probe, water, n_frames, rng, None,
                             Window(state["T"], tuple(state["stuck"])))
            classify(run.detections, node.model)
            payload = node.measure(run.detections, n_frames, run.backgrounds, flow)
            truth = sum(1 for i in run.crossing_ids if run.by_id()[i].is_plastic)
            payload = {**payload, "_true_plastic_crossing": truth,
                       "_true_per_l": water.plastic_per_ml * 1000 / node.config.concentration_factor}
        payload["_transmission_true"] = round(state["T"], 3)
        payload["_event"] = ev
        out.append(payload)
        state["T"] *= scenario.fouling_per_window  # fouling continues during the next window
        state["stuck"] = state["stuck"] + stuck_particles(rng, scenario.stuck_per_window, node.probe,
                                                          pid0=100_000 + 100 * w)
    return out
