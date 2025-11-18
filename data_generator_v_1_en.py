# ITERATION-D Synthetic Dataset Generator (Notebook-Friendly)
# Project: ITERATION-D – CAPES-STIC-AmSud

# Essential imports for random number generation
# and table (DataFrame) manipulation.
import random
import numpy as np
import pandas as pd


# ## Scenario configuration ##
# The function below, "get_scenario_config", is responsible for:
# - Creating a parameter dictionary that represents the chosen scenario.
# - Indicating whether there is a disaster, flapping, saturation, adaptive recovery, etc.
# - Defining the disaster start and end times based on the total duration.
# - Being consulted by other parts of the code to adjust metric generation.
# This is where each scenario's configuration is defined.

def get_scenario_config(scenario, duration):
    cfg = {
        "name": scenario,
        "disaster": True,
        "disaster_start": int(duration * 0.33),
        "disaster_duration": int(duration * 0.33),
        "flapping": False,
        "high_traffic": False,
        "adaptive_recovery": False,
        "stable": False,
    }

    if scenario == "C1":
        cfg["disaster"] = False
        cfg["stable"] = True

    elif scenario == "C3":
        cfg["adaptive_recovery"] = True

    elif scenario == "C4":
        cfg["high_traffic"] = True

    elif scenario == "C5":
        cfg["flapping"] = True

    if cfg["disaster"]:
        cfg["disaster_end"] = cfg["disaster_start"] + cfg["disaster_duration"]
    else:
        cfg["disaster_start"] = None
        cfg["disaster_end"] = None

    return cfg


# ## Overview of scenarios ##
#
# 1) In scenario C1 (Stable):
# - there is no disaster,
# - no flapping,
# - no congestion,
# - no adaptive recovery,
# - the network operates continuously as "normal".
#
# 2) In scenario C2 (Natural Disaster):
# - inherits the default behavior (“disaster=True”),
# - has a pre → during → post phase,
# - during the disaster we will see the largest degradations.
#
#
# 3) In scenario C3 (Adaptive Recovery):
# - same as C2, but with progressive improvement in the post-disaster phase.
#
#
# 4) In scenario C4 (Urban Saturation):
# - significantly increases queue occupancy,
# - increases loss,
# - increases jitter and delay.
#
#
# 5) In scenario C5 (Flapping):
# - links alternate up/down,
# - loss reaches 100% when the link is down,
# - latency doubles when it comes back (simulating backlog).


# ## Introduction to Part 1 of the generators ##
# In this section, we analyze how the tool builds the static network topology.
# - Nodes: edge, fog, cloud
# - Links between nodes: base latency, bandwidth and base loss
# - The scenario's temporal logic
# - How to identify if a timestamp belongs to the pre/during/post-disaster phase
# - How this function is used later in time series
# These elements are fundamental because the whole simulation is built on:
# - where the nodes are,
# - how they are connected,
# - what the physical parameters of these connections are,
# - in which simulation phase we are.

def generate_nodes(n_edge, n_fog, n_cloud):
    device_types = ["sensor", "UAV", "wearable"]
    rows = []

    for i in range(n_edge):
        rows.append([
            f"E{i+1}",      # id
            "edge",         # layer
            random.choice(device_types),  # device_type (NOT role!)
            random.choice(["urbano","rural"]),
            "low",
            1
        ])

    for i in range(n_fog):
        rows.append([
            f"F{i+1}",
            "fog",
            "gateway",
            random.choice(["urbano","rural"]),
            "medium",
            1
        ])

    for i in range(n_cloud):
        rows.append([
            f"C{i+1}",
            "cloud",
            "datacenter",
            "-",
            "high",
            1
        ])

    return pd.DataFrame(rows, columns=[
        "node_id","layer","device_type","region",
        "compute_capacity","is_critical"
    ])


# ## Explaining link logic ##
# The next step in the simulation is to connect nodes with links.
# Each link has:
# - a source (src) and a destination (dst)
# - a communication technology (LTE, WiFi, fiber, satellite)
# - a base latency dependent on the origin node's layer
# - a base bandwidth
# - a base loss probability
# These values represent physical infrastructure conditions before any disaster.
# Later, during the "during_disaster" phase, these values will be degraded according to the scenario.
#
# Allowed technologies by link type
# Origin → Destination  Possible technologies
# EDGE → FOG            LTE, WiFi, LoRaWAN
# FOG → CLOUD           fiber, microwave
# EDGE → CLOUD          satellite, LTE
#
# Base parameters by technology
# (Values inspired by real measurements of mobile networks, WiFi and fiber)
# Technology    Base latency (ms) | Base loss | Typical bandwidth (Mbps)
# LTE           30–70              0.5–5%     10–50
# WiFi          5–20               0.1–2%     50–300
# LoRaWAN       50–150             1–5%       0.02–0.05 (very low)
# fiber         1–5                <0.1%      1000–10000
# microwave     5–15               0.1–1%     200–1000
# satellite     150–700            1–10%      5–50

def generate_links_topology(nodes, n_links):
    """
    Generates a minimal but always connected topology:

    - Each EDGE node is connected to exactly 1 FOG.
    - Each FOG node is connected to at least 1 CLOUD (if clouds exist).
    - If n_links is greater than the minimum required, extra links are
      added, always respecting the pattern:
          EDGE -> FOG   or   FOG -> CLOUD.

    If n_links is smaller than the minimum needed to guarantee connectivity,
    the generator will still create all the minimum required links and, in practice,
    ignore the stricter n_links (prioritizing coherent architecture).
    """

    edges  = nodes[nodes["layer"] == "edge"]["node_id"].tolist()
    fogs   = nodes[nodes["layer"] == "fog"]["node_id"].tolist()
    clouds = nodes[nodes["layer"] == "cloud"]["node_id"].tolist()

    rows = []
    pairs = []

    # ---------------
    # Specific edge cases
    # ---------------
    if not fogs:
        # Without FOG there is not much to do in terms of E–F–C architecture.
        # We could raise an exception, but for now we return an empty DataFrame.
        return pd.DataFrame(columns=[
            "link_id","src_node_id","dst_node_id","tech",
            "base_latency_ms","base_bandwidth_mbps","base_loss_rate"
        ])

    # ---------------
    # 1) Connect each EDGE to exactly 1 FOG
    # ---------------
    for e in edges:
        f = random.choice(fogs)
        pairs.append((e, f))

    # ---------------
    # 2) Connect each FOG to at least 1 CLOUD (if clouds exist)
    # ---------------
    if clouds:
        for f in fogs:
            c = random.choice(clouds)
            pairs.append((f, c))

    # Remove duplicates while preserving order
    seen = set()
    unique_pairs = []
    for src, dst in pairs:
        if (src, dst) not in seen:
            seen.add((src, dst))
            unique_pairs.append((src, dst))
    pairs = unique_pairs

    min_required = len(pairs)

    # ---------------
    # 3) If the user requested more links (n_links > min_required),
    #    add extra valid links (EDGE–FOG, FOG–CLOUD).
    # ---------------
    # Build all possible valid pairs E–F and F–C
    all_pairs = []

    # EDGE -> FOG
    for e in edges:
        for f in fogs:
            all_pairs.append((e, f))

    # FOG -> CLOUD
    for f in fogs:
        for c in clouds:
            all_pairs.append((f, c))

    # Filter those not already in pairs
    remaining = [p for p in all_pairs if p not in seen]
    random.shuffle(remaining)

    # If n_links is smaller than min_required, prioritize min_required
    target_links = max(n_links, min_required)

    for p in remaining:
        if len(pairs) >= target_links:
            break
        pairs.append(p)
        seen.add(p)

    # ---------------
    # 4) Generate metrics for each link
    # ---------------
    rows = []
    for i, (src, dst) in enumerate(pairs, start=1):

        # Access EDGE -> FOG
        if src.startswith("E") and dst.startswith("F"):
            tech = random.choice(["LTE", "WiFi", "LoRaWAN"])
            base_latency = np.random.uniform(15, 60)
            base_bw      = np.random.uniform(5, 100)
            base_loss    = np.random.uniform(0.005, 0.05)

        # Backhaul FOG -> CLOUD
        elif src.startswith("F") and dst.startswith("C"):
            tech = random.choice(["fibra", "microondas"])
            base_latency = np.random.uniform(3, 20)
            base_bw      = np.random.uniform(200, 2000)
            base_loss    = np.random.uniform(0.0001, 0.01)

        else:
            # Fallback (should not happen in this E–F–C topology)
            tech = "LTE"
            base_latency = np.random.uniform(20, 80)
            base_bw      = np.random.uniform(5, 50)
            base_loss    = np.random.uniform(0.01, 0.1)

        rows.append([
            f"L{i}", src, dst, tech,
            base_latency, base_bw, base_loss
        ])

    return pd.DataFrame(rows, columns=[
        "link_id","src_node_id","dst_node_id","tech",
        "base_latency_ms","base_bandwidth_mbps","base_loss_rate"
    ])


# ## Temporal phases of the simulation ##
# The full simulation is divided into up to three phases:
# - pre_disaster
# - during_disaster
# - post_disaster
# The function below is used by several modules to decide whether timestamp t:
# 1) is before the disaster,
# 2) is within the disaster interval,
# or 3) is after the disaster.
# This answer determines how probabilistic models will be applied.

def phase_from_time(t, cfg):
    if not cfg["disaster"]:
        return "normal"
    if t < cfg["disaster_start"]:
        return "pre_disaster"
    if t < cfg["disaster_end"]:
        return "during_disaster"
    return "post_disaster"


def build_neighbors(links):
    neighbors = {}
    for _, row in links.iterrows():
        u = row["src_node_id"]
        v = row["dst_node_id"]
        neighbors.setdefault(u, set()).add(v)
        neighbors.setdefault(v, set()).add(u)
    return neighbors

def build_link_index(links):
    idx = {}
    for _, row in links.iterrows():
        u = row["src_node_id"]
        v = row["dst_node_id"]
        lid = row["link_id"]
        idx[(u, v)] = lid
        idx[(v, u)] = lid
    return idx

def compute_flow_path(flow, links, nodes):
    src = flow["src_node_id"]
    dst = flow["dst_node_id"]

    link_index = build_link_index(links)

    # EDGE -> FOG
    if src.startswith("E") and dst.startswith("F"):
        return link_index.get((src, dst), "")

    # EDGE -> CLOUD (via FOG)
    if src.startswith("E") and dst.startswith("C"):
        fogs = nodes[nodes["layer"] == "fog"]["node_id"].tolist()
        for f in fogs:
            if (src, f) in link_index and (f, dst) in link_index:
                return f"{link_index[(src, f)]}>{link_index[(f, dst)]}"

    return ""


# In this part of the notebook, we will analyze the function responsible for generating
# the time series of metrics for each network link.
# This time series is one of the most important parts of the dataset, as it models:
# - latency,
# - jitter,
# - loss,
# - throughput,
# - queue occupancy,
# - link up/down (is_up),
# - cause of degradation.
# The behavior of these metrics changes depending on the scenario (C1–C5) and the temporal phase (pre, during, post-disaster).

# ## Purpose of generate_link_timeseries() ##
# This function:
# - receives the links table (static topology),
# - applies rules and statistical distributions for each timestamp,
# - generates disaster effects, flapping, congestion, etc.,
# - returns a long table with one row per (link × timestamp).
# The output has dozens, hundreds or thousands of rows depending on duration and step.

def generate_link_timeseries(links, cfg, duration, step):
    """
    Generates time series for links.

    Important corrections:
    - latency and jitter never become negative (physical truncation);
    - all degradation (disaster, flapping, congestion) is applied
      directly on links, which will be the basis for flows.
    """

    timestamps = np.arange(0, duration, step)
    rows = []

    for _, link in links.iterrows():
        for t in timestamps:

            phase = phase_from_time(t, cfg)

            # Base state (network "normal")
            latency = max(0.1, np.random.normal(link["base_latency_ms"], 3))
            jitter  = max(0.1, np.random.normal(2, 1))
            loss    = np.random.uniform(0, link["base_loss_rate"])
            is_up   = 1
            cause   = "none"

            # -------- Disaster phase --------
            if not cfg["stable"]:
                if phase == "during_disaster":
                    latency *= np.random.uniform(1.5, 3.5)
                    jitter  *= np.random.uniform(1.2, 1.8)
                    loss    += np.random.uniform(0.05, 0.25)
                    cause    = "infra_danificada"

                    # Flapping (C5)
                    if cfg["flapping"]:
                        if (t // (step * 3)) % 2 == 0 and random.random() < 0.3:
                            is_up = 0
                            loss  = 1.0
                            latency *= 2
                            jitter  *= 2
                    else:
                        # Occasional link drop (C2)
                        if random.random() < 0.1:
                            is_up = 0
                            loss  = 1.0

                # -------- Post-disaster --------
                elif phase == "post_disaster":
                    if cfg["adaptive_recovery"]:  # C3
                        latency *= np.random.uniform(0.8, 1.2)
                        jitter  *= np.random.uniform(0.8, 1.2)
                        loss = max(0.0, loss - np.random.uniform(0.02, 0.08))
                        cause = "recuperacao"
                    else:
                        latency *= np.random.uniform(0.9, 1.3)
                        jitter  *= np.random.uniform(0.9, 1.4)
                        loss = min(1.0, loss + np.random.uniform(0.0, 0.05))

            # -------- Congestion (C4) --------
            if cfg["high_traffic"]:
                loss   += np.random.uniform(0.02, 0.12)
                jitter *= np.random.uniform(1.1, 1.5)
                if phase != "pre_disaster":
                    cause = "congestionamento"

            # Final normalizations
            latency = max(0.1, latency)
            jitter  = max(0.1, jitter)
            loss    = min(max(loss, 0.0), 1.0)

            throughput = (
                np.random.uniform(0.5, 1.0)
                * link["base_bandwidth_mbps"]
                * (1 if is_up else 0)
            )

            queue_occ = (
                np.random.uniform(0.3, 1.0)
                if cfg["high_traffic"]
                else np.random.uniform(0.0, 0.3)
            )

            rows.append([
                t, link["link_id"], phase,
                latency, jitter, loss,
                throughput, queue_occ,
                is_up, cause
            ])

    return pd.DataFrame(rows, columns=[
        "timestamp","link_id","phase",
        "latency_ms","jitter_ms","loss_rate",
        "throughput_mbps","queue_occupancy",
        "is_up","degradation_cause"
    ])


# ## Introduction to the next generators: ##
# In this stage we study three essential generators:
# 1. generate_flows()
# Responsible for creating logical flows between network nodes, modeled as:
# - critical telemetry,
# - actuator control,
# - drone video,
# - public alerts,
# - batch logs,
# - best-effort.
# These flows have different requirements: latency, reliability, priority and traffic patterns.
# 2. generate_events()
# Simulates external events such as:
# - earthquake,
# - aftershock,
# - individual link drops,
# - flapping.
# Events are fundamental for scenarios C2–C5.
# 3. generate_control_actions()
# Simulates actions taken by the system (fog or cloud) to restore SLA, such as:
# - rerouting flows,
# - changing priority,
# - reducing load (throttle),
# - migrating services.

def generate_flows(nodes, links, cfg, duration):
    app_profiles = {
        "telemetria_critica": ("telemetria_critica", 1, 150, 0.99),
        "video_drone":        ("video_drone",        2, 300, 0.95),
        "best_effort":        ("best_effort",        3, 800, 0.90),
    }

    fogs = nodes[nodes["layer"]=="fog"]["node_id"].tolist()
    clouds = nodes[nodes["layer"]=="cloud"]["node_id"].tolist()
    neighbors = build_neighbors(links)

    rows = []
    fid = 1

    for _, row in nodes[nodes["layer"]=="edge"].iterrows():
        src = row["node_id"]
        dtype = row["device_type"]

        fog_neighbors = [n for n in neighbors.get(src,[]) if n in fogs]

        cloud_candidates = []
        for f in fog_neighbors:
            for c in neighbors.get(f,[]):
                if c in clouds:
                    cloud_candidates.append((f,c))

        if dtype == "sensor" and fog_neighbors:
            app,prio,lat,rel = app_profiles["telemetria_critica"]
            dst = fog_neighbors[0]
            rows.append([f"FL{fid}",src,dst,app,prio,lat,rel,"constante",0,duration])
            fid+=1

        elif dtype == "UAV" and (fog_neighbors or cloud_candidates):
            app,prio,lat,rel = app_profiles["video_drone"]
            if fog_neighbors and (not cloud_candidates or random.random()<0.5):
                dst = fog_neighbors[0]
            else:
                _,dst = cloud_candidates[0]
            rows.append([f"FL{fid}",src,dst,app,prio,lat,rel,"bursty",0,duration])
            fid+=1

        elif dtype == "wearable" and cloud_candidates:
            app,prio,lat,rel = app_profiles["best_effort"]
            _,dst = cloud_candidates[0]
            rows.append([f"FL{fid}",src,dst,app,prio,lat,rel,"on_off",0,duration])
            fid+=1

    flows = pd.DataFrame(rows, columns=[
        "flow_id","src_node_id","dst_node_id","app_type","priority",
        "required_latency_ms","required_reliability",
        "traffic_pattern","start_time","end_time"
    ])

    if cfg["high_traffic"]:
        flows = pd.concat([flows, flows.sample(frac=1, replace=True)], ignore_index=True)

    return flows


# ## Explaining the events generator ##
# Events represent external occurrences that affect the network.
# In ITERATION-D, we consider:
# - earthquake (global impact),
# - aftershock (global but smaller impact),
# - individual link drops,
# - flapping in specific scenarios.
# These events serve as triggers for degradation and control actions.

def generate_events(links, cfg, duration):
    rows = []
    eid = 1

    if cfg["disaster"]:
        rows.append([f"E{eid}", cfg["disaster_start"] - 10,
                     "earthquake_shock", "region", "global", 5])
        eid += 1

        rows.append([f"E{eid}", cfg["disaster_start"] + 300,
                     "aftershock", "region", "global", 3])
        eid += 1

        affected = random.sample(links["link_id"].tolist(),
                                 max(1, len(links)//5))

        for lid in affected:
            t = random.randint(cfg["disaster_start"], cfg["disaster_end"])
            rows.append([f"E{eid}", t, "link_down", "link", lid, 4])
            eid += 1

        if cfg["flapping"]:
            for lid in affected[: len(affected)//2 ]:
                t = random.randint(cfg["disaster_start"], cfg["disaster_end"])
                rows.append([f"E{eid}", t, "link_flapping", "link", lid, 2])
                eid += 1

    return pd.DataFrame(rows, columns=[
        "event_id","timestamp","type","target_type","target_id","severity"
    ])


# ## The control actions generator ##
# The function generate_control_actions() simulates what the system tries to do to restore SLA after the disaster:
# - reroute_flow → change path
# - promote_flow_priority → increase priority
# - throttle_video → reduce video traffic
# - migrate_service → move service to another node
#
# These actions represent some adaptive strategies that can be implemented.

def generate_control_actions(flows, cfg):
    rows = []
    aid = 1

    if not cfg["disaster"]:
        return pd.DataFrame(columns=[
            "action_id","timestamp","controller_layer",
            "action_type","target_id","expected_effect",
            "observed_effect_sla"
        ])

    sample_flows = flows["flow_id"].tolist()[:10]

    for f in sample_flows:
        rows.append([
            f"A{aid}",
            cfg["disaster_start"] + cfg["disaster_duration"]//2,
            random.choice(["fog", "cloud"]),
            random.choice(["reroute_flow","promote_flow_priority",
                           "throttle_video","migrate_service"]),
            f,
            "restore_SLA",
            random.choice(["improved", "neutral"])
        ])
        aid += 1

    return pd.DataFrame(rows, columns=[
        "action_id","timestamp","controller_layer","action_type",
        "target_id","expected_effect","observed_effect_sla"
    ])


# In this part, we study the highest-level simulator function: generate_flow_timeseries()
# It produces, for each network flow:
# - end-to-end latency,
# - jitter,
# - bytes delivered,
# - packets dropped,
# - SLA compliance,
# - logical path (path_links),
# - temporal phase of the simulation.
#
# Unlike the link time series, which works per-link, this function works per-flow, integrating:
#
# - application behavior,
# - SLA requirements,
# - disaster impact,
# - accumulated degradation,
# - possible congestion or flapping effects.

def generate_flow_timeseries(flows, links, nodes, link_timeseries, cfg, duration, step):
    """
    Generates time series for flows derived from the links.

    - path_links comes from compute_flow_path (e.g. "L1>L3");
    - for each timestamp t, we fetch the metrics of the links in the path from link_timeseries;
    - end-to-end latency  = sum of link latencies;
    - end-to-end jitter   = sum of link jitters;
    - end-to-end loss     = 1 - ∏ (1 - loss_i);
    - throughput_mbps     = min(throughput_mbps of links)  (bottleneck);
    - delivered/dropped   = derived from throughput + loss.

    Output includes both throughput in Mbps and bytes per interval.
    """

    timestamps = np.arange(0, duration, step)

    # Precompute each flow's path
    flow_paths_str = {}
    flow_paths_list = {}

    for _, flow in flows.iterrows():
        path_str = compute_flow_path(flow, links, nodes)  # e.g. "L1>L3"
        flow_paths_str[flow["flow_id"]] = path_str
        if path_str:
            flow_paths_list[flow["flow_id"]] = path_str.split(">")
        else:
            flow_paths_list[flow["flow_id"]] = []

    rows = []

    for _, flow in flows.iterrows():
        flow_id = flow["flow_id"]
        path_links = flow_paths_list[flow_id]

        # If the flow has no valid path, skip it
        if not path_links:
            continue

        required_lat = flow["required_latency_ms"]
        required_rel = flow["required_reliability"]

        for t in timestamps:

            if not (flow["start_time"] <= t <= flow["end_time"]):
                continue

            phase = phase_from_time(t, cfg)

            lat_list  = []
            jit_list  = []
            loss_list = []
            thr_list  = []

            # Collect metrics for each link in the path
            for lid in path_links:
                link_row = link_timeseries[
                    (link_timeseries["link_id"] == lid) &
                    (link_timeseries["timestamp"] == t)
                ]

                if link_row.empty:
                    continue

                r = link_row.iloc[0]
                lat_list.append(r["latency_ms"])
                jit_list.append(r["jitter_ms"])
                loss_list.append(r["loss_rate"])
                thr_list.append(r["throughput_mbps"])

            # If we didn't find any link data for this t, ignore
            if not lat_list:
                continue

            # -------------------------
            # End-to-end composition
            # -------------------------

            # Latency/jitter: sum
            lat_e2e = sum(lat_list)
            jit_e2e = sum(jit_list)

            # Composed loss: 1 - ∏(1 - loss_i)
            success_prob = 1.0
            for li in loss_list:
                success_prob *= (1.0 - li)
            loss_e2e = 1.0 - success_prob
            loss_e2e = min(max(loss_e2e, 0.0), 1.0)

            # End-to-end throughput (path bottleneck)
            if thr_list:
                thr_e2e_mbps = max(0.0, min(thr_list))
            else:
                thr_e2e_mbps = 0.0

            # Bytes delivered/dropped in the interval
            # throughput (Mbps) -> bytes/s -> bytes/interval
            # 1 Mbps = 1e6 bits/s = 1e6/8 bytes/s

            bytes_per_sec = (thr_e2e_mbps * 1e6) / 8.0
            interval_seconds = step

            total_bytes_interval = bytes_per_sec * interval_seconds

            delivered_bytes = total_bytes_interval * (1.0 - loss_e2e)
            dropped_bytes   = total_bytes_interval * loss_e2e

            # Keep values "clean"
            delivered_bytes = max(0.0, delivered_bytes)
            dropped_bytes   = max(0.0, dropped_bytes)

            # -------------------------
            # SLA
            # -------------------------
            sla_met = int(
                (lat_e2e <= required_lat) and
                ((1.0 - loss_e2e) >= required_rel)
            )

            rows.append([
                t,
                flow_id,
                flow_paths_str[flow_id],    # "L1>L3"
                lat_e2e,
                jit_e2e,
                thr_e2e_mbps,
                delivered_bytes,
                dropped_bytes,
                sla_met,
                phase
            ])

    return pd.DataFrame(rows, columns=[
        "timestamp",
        "flow_id",
        "path_links",
        "end_to_end_latency_ms",
        "end_to_end_jitter_ms",
        "throughput_mbps",            # E2E throughput for the flow
        "delivered_bytes_interval",   # bytes effectively delivered in the interval
        "dropped_bytes_interval",     # bytes "lost" in the interval
        "sla_met",
        "phase"
    ])


# ## The role of run_generator ##
# So far we've seen several blocks of the tool:
# - get_scenario_config → defines the scenario's global behavior (C1–C5).
# - generate_nodes → creates the node topology (edge, fog, cloud).
# - generate_links (or generate_links_realistic) → creates links with base latency, loss and bandwidth.
# - generate_link_timeseries → generates per-link network metrics over time.
# - generate_flows → creates logical flows (applications).
# - generate_events → introduces earthquake, aftershocks, link drops, flapping.
# - generate_control_actions → simulates control actions (fog/cloud).
# - generate_flow_timeseries → generates end-to-end metrics per flow.
# The run_generator function is the main orchestrator:
# It calls all other functions in the correct order.
# Ensures reproducibility (via seed).
# Returns everything in a single dictionary of DataFrames — ideal for Jupyter.

def run_generator(
    scenario,
    n_edge,
    n_fog,
    n_cloud,
    n_links,
    duration,
    step,
    seed
):
    """
    Generates all DataFrames for the ITERATION-D scenario and returns them in a dictionary.
    """

    random.seed(seed)
    np.random.seed(seed)

    cfg = get_scenario_config(scenario, duration)

    # Topology
    nodes = generate_nodes(n_edge, n_fog, n_cloud)
    links = generate_links_topology(nodes, n_links)

    # Flows (defined based on topology)
    flows = generate_flows(nodes, links, cfg, duration)

    # Link time series
    link_ts = generate_link_timeseries(links, cfg, duration, step)

    # Events and actions
    events  = generate_events(links, cfg, duration)
    actions = generate_control_actions(flows, cfg)

    # Flow time series — now based on link_ts
    flow_ts = generate_flow_timeseries(flows, links, nodes, link_ts, cfg, duration, step)

    return {
        "nodes": nodes,
        "links": links,
        "flows": flows,
        "link_timeseries": link_ts,
        "flow_timeseries": flow_ts,
        "events": events,
        "control_actions": actions,
    }


# Now that all internal functions are defined, let's finally run the full generator using run_generator().
#
# In the block below:
# - We run the simulation for a chosen scenario (C1–C5).
# - We receive the seven generated DataFrames:
#     - nodes
#     - links
#     - link_timeseries
#     - flows
#     - flow_timeseries
#     - events
#     - control_actions
#
# We save all these DataFrames as .csv files with standardized names to ease later processing and analysis.

# Execute the synthetic data generator for the chosen scenario

scenario = "C2"

data = run_generator(
    scenario=scenario,   # Simulation scenario (C1=stable, C2=disaster, C3=recovery,
                     # C4=urban saturation, C5=flapping)

    n_edge=20,       # Number of devices in the EDGE layer
    n_fog=5,         # Number of intermediate nodes in the FOG layer
    n_cloud=1,       # Number of datacenters in the CLOUD layer

    n_links=25,      # Total number of links to be generated

    duration=3600,   # Total simulation duration (in seconds)
    step=5,         # Time step between points in the time series (seconds)

    seed=123         # Seed for reproducibility
)

# Automatic saving of all generated files
# Each key in the 'data' dictionary becomes a CSV file.

for name, df in data.items():

    # The filename follows the pattern:
    #   iterationD_<dataframe name>_<scenario>.csv
    filename = f"iterationD_{name}_{scenario}.csv"

    # Export the DataFrame to CSV
    df.to_csv(filename, index=False)

    # Informational log
    print("Arquivo salvo:", filename)
