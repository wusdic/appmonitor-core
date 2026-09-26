"""Synthetic observation source.

In production, Observations come from capture adapters (SPAN/TAP decode, flow
collector, active prober). For development and demos this generator fabricates
realistic multi-system, multi-entity traffic with time-of-day modulation, plus
scheduled anomalies and a behavioural-drift scenario, so every downstream
engine has something to chew on and the detections are demonstrable.

It emits per-tick batches of Observation objects on a *virtual* clock, so a
warm-up run can synthesise days of history in seconds to seed baselines.
"""
from __future__ import annotations

import math
import random
import string
from dataclasses import dataclass
from typing import Dict, List

from ..models.schema import AcquisitionMethod, Observation, Reachability


def _rand_label(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(n))


@dataclass
class Persona:
    entity: str
    archetype: str


class TrafficGenerator:
    """Deterministic-ish multi-system traffic fabricator."""

    def __init__(self, seed: int = 42, window_s: int = 60):
        self.rng = random.Random(seed)
        self.window_s = window_s
        self.vt = 0.0                       # virtual clock (epoch seconds)
        self.tick = 0
        self.systems: Dict[str, List[Persona]] = {
            "erp-prod": [
                Persona("10.20.1.11", "interactive"),
                Persona("10.20.1.12", "interactive"),
                Persona("10.20.1.13", "interactive"),
                Persona("10.20.4.30", "api"),
                Persona("10.20.4.31", "integration"),
                Persona("10.20.9.5", "backup"),
                Persona("10.20.9.9", "health"),
            ],
            "oa-portal": [
                Persona("10.30.2.21", "interactive"),
                Persona("10.30.2.22", "interactive"),
                Persona("10.30.2.23", "search"),
                Persona("10.30.4.40", "api"),
                Persona("10.30.9.9", "health"),
            ],
            "api-gateway": [
                Persona("10.40.4.51", "api"),
                Persona("10.40.4.52", "api"),
                Persona("10.40.4.53", "integration"),
                Persona("10.40.9.9", "health"),
                Persona("10.40.9.6", "backup"),
            ],
        }
        # anomalies injected on the LIVE timeline (tick index into live run)
        self.anomaly_schedule = [
            {"after_tick": 3, "system": "erp-prod", "entity": "10.20.1.13",
             "mode": "exfil", "note": "interactive user turns to bulk upload (drift + exfil)"},
            {"after_tick": 5, "system": "oa-portal", "entity": "10.30.2.99",
             "mode": "scanner", "note": "new host runs port/host scan"},
            {"after_tick": 7, "system": "api-gateway", "entity": "10.40.4.52",
             "mode": "beacon", "note": "api client shifts to periodic beacon"},
            {"after_tick": 9, "system": "erp-prod", "entity": "10.20.7.77",
             "mode": "dns_tunnel", "note": "host starts DNS tunnelling"},
            {"after_tick": 11, "system": "oa-portal", "entity": "10.30.2.21",
             "mode": "bruteforce", "note": "account credential stuffing"},
        ]
        self._overrides: Dict[str, str] = {}     # (system|entity) -> mode

    # ------------------------------------------------------------------- step
    def step(self, dt: float, live: bool = False) -> List[Observation]:
        self.vt += dt
        now = self.vt
        if live:
            for a in self.anomaly_schedule:
                if self.tick >= a["after_tick"]:
                    self._overrides[f"{a['system']}|{a['entity']}"] = a["mode"]

        hour = int((now / 3600) % 24)
        biz = self._business_factor(hour)
        obs: List[Observation] = []

        # modes that mean "the host has been repurposed" — its normal business
        # traffic stops and only the anomalous activity remains (crisp drift).
        replace_modes = {"scanner", "dns_tunnel", "bruteforce", "beacon"}
        for system, personas in self.systems.items():
            existing = {p.entity for p in personas}
            for p in personas:
                mode = self._overrides.get(f"{system}|{p.entity}")
                if mode in replace_modes:
                    obs.extend(self._emit_override(system, p.entity, mode, now, self.rng))
                    continue
                obs.extend(self._emit(system, p.entity, p.archetype, now, biz, self.rng))
                if mode:                       # additive modes (e.g. exfil)
                    obs.extend(self._emit_override(system, p.entity, mode, now, self.rng))
            # brand-new injected entities that are not real personas
            for a in self.anomaly_schedule:
                if a["system"] == system and a["entity"] not in existing \
                        and self._overrides.get(f"{system}|{a['entity']}"):
                    obs.extend(self._emit_override(system, a["entity"],
                                                   self._overrides[f"{system}|{a['entity']}"],
                                                   now, self.rng))
        if live:
            self.tick += 1
        return obs

    @staticmethod
    def _business_factor(hour: int) -> float:
        if 6 <= hour <= 20:
            return 0.2 + 0.8 * max(0.0, math.sin(math.pi * (hour - 6) / 14))
        return 0.15

    # ------------------------------------------------------- helper builders
    def _http(self, system, entity, now, method, host, path, status, up, down, dur,
              rng, sni=None, ua="Mozilla/5.0", ctype="text/html", dport=443):
        return Observation(
            ts=now, system=system, entity=entity, peer=host,
            method=AcquisitionMethod.PASSIVE_SPAN, l3_proto="ip", l4_proto="tcp",
            dst_port=dport, bytes_up=up, bytes_down=down,
            pkts_up=max(1, up // 500), pkts_down=max(1, down // 1400),
            rtt_ms=rng.uniform(5, 40), win_size=64240, duration_ms=dur,
            app_proto="http", http_method=method, http_host=host, http_path=path,
            http_status=status, user_agent=ua, content_type=ctype,
            tls_version="TLS1.3", tls_cipher="TLS_AES_128_GCM_SHA256",
            tls_sni=sni or host, ja3="769,47-53,0-11", ja3s="771,49200")

    def _dns(self, system, entity, now, qname, rng, qtype="A", rcode="NOERROR"):
        return Observation(
            ts=now, system=system, entity=entity, peer="10.0.0.53", l4_proto="udp",
            dst_port=53, app_proto="dns", dns_qname=qname, dns_qtype=qtype,
            dns_rcode=rcode, bytes_up=60 + len(qname), bytes_down=120,
            pkts_up=1, pkts_down=1, duration_ms=rng.uniform(1, 10))

    def _probe(self, system, entity, now, rng, reach=Reachability.REACHABLE):
        return Observation(
            ts=now, system=system, entity=entity, peer=entity,
            method=AcquisitionMethod.ACTIVE_PROBE, reachability=reach,
            rtt_ms=rng.uniform(2, 30) if reach == Reachability.REACHABLE else 0.0,
            hop_count=rng.randint(3, 9), open_ports=(443, 80) if reach == Reachability.REACHABLE else ())

    # ---------------------------------------------------- archetype emitters
    def _emit(self, system, entity, archetype, now, biz, rng) -> List[Observation]:
        obs: List[Observation] = []
        if archetype == "interactive":
            host = {"erp-prod": "erp.corp.local", "oa-portal": "portal.corp.local"}.get(system, "app.corp.local")
            n = max(0, int(rng.gauss(9, 3) * biz))
            paths = ["/home", "/orders", "/orders/view", "/report", "/profile",
                     "/search", "/inbox", "/dashboard", "/item/123", "/settings"]
            for _ in range(n):
                method = "GET" if rng.random() > 0.15 else "POST"
                status = 200 if rng.random() > 0.05 else rng.choice([302, 404])
                obs.append(self._http(system, entity, now, method, host, rng.choice(paths),
                                      status, rng.randint(200, 1200), rng.randint(2000, 40000),
                                      rng.uniform(50, 400), rng))
            for _ in range(max(1, n // 4)):
                obs.append(self._dns(system, entity, now, rng.choice(
                    [host, "cdn.corp.local", "auth.corp.local"]), rng))
        elif archetype == "search":
            n = max(0, int(rng.gauss(16, 4) * biz))
            for _ in range(n):
                obs.append(self._http(system, entity, now, "GET", "portal.corp.local",
                                      "/search", 200, rng.randint(300, 900),
                                      rng.randint(1500, 12000), rng.uniform(30, 200), rng))
        elif archetype == "api":
            base = max(2, int(rng.gauss(22, 4) * (0.5 + 0.5 * biz)))
            for _ in range(base):
                status = 200 if rng.random() > 0.03 else 500
                obs.append(self._http(system, entity, now, "GET" if rng.random() > 0.4 else "POST",
                                      "api.corp.local", "/v1/resource", status,
                                      rng.randint(300, 800), rng.randint(500, 6000),
                                      rng.uniform(10, 60), rng, ctype="application/json"))
        elif archetype == "integration":
            for _ in range(max(2, int(rng.gauss(18, 3)))):
                obs.append(self._http(system, entity, now, "POST", "svc.corp.local",
                                      "/api/sync", 200, rng.randint(800, 3000),
                                      rng.randint(800, 4000), rng.uniform(15, 70), rng,
                                      ctype="application/json"))
        elif archetype == "backup":
            for _ in range(max(1, int(rng.gauss(4, 1)))):
                obs.append(self._http(system, entity, now, "PUT", "backup.corp.local",
                                      "/blob", 200, rng.randint(2_000_000, 6_000_000),
                                      rng.randint(500, 2000), rng.uniform(500, 3000), rng,
                                      ctype="application/octet-stream"))
        elif archetype == "health":
            obs.append(self._http(system, entity, now, "GET", "svc.corp.local",
                                  "/healthz", 200, 120, 800, rng.uniform(3, 12), rng))
            obs.append(self._probe(system, entity, now, rng))
        return obs

    # ---------------------------------------------------------- anomaly modes
    def _emit_override(self, system, entity, mode, now, rng) -> List[Observation]:
        obs: List[Observation] = []
        if mode == "exfil":
            for _ in range(6):
                obs.append(self._http(system, entity, now, "POST", "ext-store.example.net",
                                      "/upload", 200, rng.randint(3_000_000, 8_000_000),
                                      rng.randint(500, 1500), rng.uniform(800, 4000), rng,
                                      sni="ext-store.example.net", ctype="application/octet-stream"))
        elif mode == "scanner":
            for _ in range(60):
                obs.append(Observation(
                    ts=now, system=system, entity=entity,
                    peer=f"10.{rng.randint(20,40)}.{rng.randint(1,9)}.{rng.randint(2,254)}",
                    l4_proto="tcp", dst_port=rng.randint(1, 9000), tcp_flags="SYN",
                    bytes_up=60, bytes_down=0, pkts_up=1, pkts_down=0,
                    duration_ms=rng.uniform(1, 50), rtt_ms=rng.uniform(1, 20)))
            for _ in range(24):
                obs.append(self._http(system, entity, now, "GET", "app.corp.local",
                                      f"/{_rand_label(rng, 8)}", 404, 80, 300, rng.uniform(5, 40), rng))
        elif mode == "beacon":
            for _ in range(3):
                obs.append(self._http(system, entity, now, "GET", "c2.example.net",
                                      "/ping", 200, 120, 800, rng.uniform(10, 15), rng,
                                      sni="c2.example.net"))
        elif mode == "dns_tunnel":
            for _ in range(25):
                obs.append(self._dns(system, entity, now,
                                     f"{_rand_label(rng, 48)}.tunnel.example.net", rng,
                                     qtype="TXT", rcode=rng.choice(["NOERROR", "NXDOMAIN"])))
        elif mode == "bruteforce":
            for _ in range(50):
                status = 401 if rng.random() > 0.1 else 200
                obs.append(self._http(system, entity, now, "POST", "portal.corp.local",
                                      "/login", status, 400, 600, rng.uniform(20, 80), rng))
        return obs
