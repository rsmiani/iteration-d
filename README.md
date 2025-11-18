
---

# **README**

# ITERATION-D Synthetic Dataset Generator
### CAPES–STIC–AMSud International Project  
### Brazil • Uruguay • Chile • France

This repository contains the **synthetic network dataset generator** used in the **ITERATION-D – CAPES–STIC-AMSud** project.  
Its goal is to simulate **Edge–Fog–Cloud architectures** under normal and disaster-related scenarios, creating comprehensive datasets for research on resilience, monitoring, anomaly detection, and distributed decision-making.

---

## Goals

The generator produces realistic datasets for evaluating:

- distributed network behavior (edge/fog/cloud),
- link-level metrics (latency, jitter, loss, throughput),
- end-to-end flow performance,
- disaster impact on communication,
- control and mitigation actions executed by Fog/Cloud layers,
- path reconstruction and inference,
- SLA preservation and resilience mechanisms.

---

## Simulated Architecture

Three-layer network:

1. **Edge**  
   Device types:  
   - `sensor` → critical telemetry  
   - `UAV` → video stream  
   - `wearable` → best-effort traffic  

2. **Fog**  
   Intermediate gateways  

3. **Cloud**  
   Regional datacenter  

The topology is always coherent:
- each Edge → exactly 1 Fog  
- each Fog → at least 1 Cloud  
- optional additional links follow only Edge→Fog or Fog→Cloud  

---

## Available Scenarios

| Scenario | Description |
|---------|-------------|
| **C1** | Stable network (no disaster) - OK|
| **C2** | Standard disaster - OK|
| **C3** | Disaster with adaptive recovery - Soon|
| **C4** | Disaster with urban congestion - Soon|
| **C5** | Disaster with link flapping - Soon|

Each scenario changes link behavior and thus affects flow performance.

---

## Output Files (CSV)

### **1. nodes.csv**
Structure of devices:
- id, layer, device_type, region, capacity, criticality

### **2. links.csv**
Physical network graph:
- src → dst  
- technology  
- base latency/bandwidth/loss  

### **3. link_timeseries.csv**
Time-series metrics:
- latency, jitter, loss  
- throughput (Mbps)  
- queue occupancy  
- up/down state  
- degradation cause  

### **4. flows.csv**
End-to-end flows:
- application type  
- SLA requirements  

### **5. flow_timeseries.csv**
End-to-end time series:
- latency = sum(latencies of each link)  
- jitter = sum(jitters)  
- loss = composed product  
- throughput = bottleneck link  
- delivered / dropped bytes  

### **6. events.csv**
Disaster events: earthquake, aftershock, link_down, flapping - TBD

### **7. control_actions.csv**
Mitigation actions: reroute, promote, throttle, migrate - TBD

---

## Usage (Python Script)

```python
from iterationD_generator import run_generator

data = run_generator(
    scenario="C2",
    n_edge=20,
    n_fog=5,
    n_cloud=1,
    n_links=30,
    duration=1800,
    step=10,
    seed=123
)

for name, df in data.items():
    df.to_csv(f"{name}.csv", index=False)
