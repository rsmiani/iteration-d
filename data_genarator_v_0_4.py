#!/usr/bin/env python
# coding: utf-8

# In[29]:


# ITERATION-D Synthetic Dataset Generator (Notebook-Friendly)
# Project: ITERATION-D – CAPES-STIC-AmSud

# Importações essenciais para geração de números aleatórios
# e manipulação de tabelas (DataFrames).
import random
import numpy as np
import pandas as pd


# ## Configuração dos cenários ##
# A função abaixo, "get_scenario_config", é responsável por:
# - Criar um dicionário de parâmetros que representa o cenário escolhido.
# - Indicar se há desastre, flapping, saturação, recuperação adaptativa etc.
# - Definir os tempos de início e fim do desastre com base na duração total.
# - Ser consultada pelas demais partes do código para ajustar a geração das métricas.
# É aqui que as configurações de cada cenário são definidas.

# In[30]:


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


# ## Visão geral dos cenários ##
# 
# 1) No cenário C1 (Estável):
# não há desastre,
# não há flapping,
# não há congestionamento,
# não há recuperação adaptativa,
# a rede opera continuamente em "normal".
# 
# 2) No cenário C2 (Desastre Natural):
# herda o comportamento padrão (“disaster=True”),
# tem uma fase pre → during → post,
# durante o desastre veremos as maiores degradações.
# 
# 
# 3) No cenário C3 (Recuperação Adaptativa):
# igual ao C2, mas com melhoria progressiva no post-disaster.
# 
# 
# 4) No cenário C4 (Saturação Urbana):
# aumenta significativamente a ocupação de fila,
# aumenta perda
# aumenta jitter e atraso.
# 
# 
# 5) No cenário C5 (Flapping):
# links alternam up/down,
# perda atinge 100% quando o link está down,
# latência dobra ao voltar (simulando backlog).

# ## Introdução à Parte 1 dos Geradores ##
# Nesta seção, vamos analisar como a ferramenta constrói a topologia estática da rede.
# - Nós: edge, fog, cloud
# - Enlaces entre nós: latência base, banda e perda base
# - A lógica temporal do cenário
# - Como identificar se um instante pertence à fase pre/during/post-disaster
# - Como essa função é utilizada depois nas séries temporais
# Esses elementos são fundamentais, pois toda a simulação é construída sobre:
# - onde estão os nós,
# - como eles se conectam,
# - quais são os parâmetros físicos dessas conexões,
# - em que fase da simulação nos encontramos.

# In[31]:


def generate_nodes(n_edge, n_fog, n_cloud):
    device_types = ["sensor", "UAV", "wearable"]
    rows = []

    for i in range(n_edge):
        rows.append([
            f"E{i+1}",      # id
            "edge",         # layer
            random.choice(device_types),  # device_type (NÃO É role!)
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


# ## Explicando a lógica de links ##
# O próximo passo da simulação é conectar os nós entre si através de enlaces.
# Cada link possui:
# - uma origem (src) e um destino (dst)
# - uma tecnologia de comunicação (LTE, WiFi, fibra, satélite)
# - uma latência base dependente da camada do nó de origem
# - uma banda base
# - uma probabilidade de perda base
# Esses valores representam condições físicas da infraestrutura antes de qualquer desastre.
# Depois, durante a fase "during_disaster", esses valores serão degradados conforme o cenário.
# 
# Tecnologias permitidas por tipo de link
# Origem → Destino	Tecnologias possíveis
# EDGE → FOG	LTE, WiFi, LoRaWAN
# FOG → CLOUD	fibra, micro-ondas
# EDGE → CLOUD	satélite, LTE
# 
# Parâmetros base por tecnologia
# (Valores inspirados em medições reais de redes móveis, WiFi e fibra)
# Tecnologia Latência base (ms)	Perda base Banda típica (Mbps)
# LTE	30–70	0.5–5%	10–50
# WiFi	5–20	0.1–2%	50–300
# LoRaWAN	50–150	1–5%	0.02–0.05 (muito baixa)
# fibra	1–5	<0.1%	1000–10000
# micro-ondas	5–15	0.1–1%	200–1000
# satélite	150–700	1–10%	5–50

# In[32]:


def generate_links_topology(nodes, n_links):
    """
    Gera uma topologia minimalista porém sempre conectada:

    - Cada nó EDGE é conectado a exatamente 1 FOG.
    - Cada nó FOG é conectado a pelo menos 1 CLOUD (se clouds existirem).
    - Se n_links for maior que o mínimo necessário, links extras são
      adicionados, sempre respeitando o padrão:
          EDGE -> FOG   ou   FOG -> CLOUD.

    Se n_links for menor que o mínimo necessário para garantir conectividade,
    o gerador ainda criará todos os links mínimos necessários e, na prática,
    ignorará o n_links mais restritivo (priorizando a arquitetura coerente).
    """

    edges  = nodes[nodes["layer"] == "edge"]["node_id"].tolist()
    fogs   = nodes[nodes["layer"] == "fog"]["node_id"].tolist()
    clouds = nodes[nodes["layer"] == "cloud"]["node_id"].tolist()

    rows = []
    pairs = []

    # ---------------
    # Casos degenerados
    # ---------------
    if not fogs:
        # Sem FOG não há muito o que fazer em termos de arquitetura E–F–C.
        # Poderíamos lançar uma exceção, mas por ora retornamos DataFrame vazio.
        return pd.DataFrame(columns=[
            "link_id","src_node_id","dst_node_id","tech",
            "base_latency_ms","base_bandwidth_mbps","base_loss_rate"
        ])

    # ---------------
    # 1) Conectar cada EDGE a exatamente 1 FOG
    # ---------------
    for e in edges:
        f = random.choice(fogs)
        pairs.append((e, f))

    # ---------------
    # 2) Conectar cada FOG a pelo menos 1 CLOUD (se existirem clouds)
    # ---------------
    if clouds:
        for f in fogs:
            c = random.choice(clouds)
            pairs.append((f, c))

    # Remover duplicatas mantendo a ordem
    seen = set()
    unique_pairs = []
    for src, dst in pairs:
        if (src, dst) not in seen:
            seen.add((src, dst))
            unique_pairs.append((src, dst))
    pairs = unique_pairs

    min_required = len(pairs)

    # ---------------
    # 3) Se o usuário pediu mais links (n_links > min_required),
    #    adicionamos links extras válidos (EDGE–FOG, FOG–CLOUD).
    # ---------------
    # Construir todos os pares válidos possíveis E–F e F–C
    all_pairs = []

    # EDGE -> FOG
    for e in edges:
        for f in fogs:
            all_pairs.append((e, f))

    # FOG -> CLOUD
    for f in fogs:
        for c in clouds:
            all_pairs.append((f, c))

    # Filtrar os que ainda não estão em pairs
    remaining = [p for p in all_pairs if p not in seen]
    random.shuffle(remaining)

    # Se n_links for menor que min_required, priorizamos min_required
    target_links = max(n_links, min_required)

    for p in remaining:
        if len(pairs) >= target_links:
            break
        pairs.append(p)
        seen.add(p)

    # ---------------
    # 4) Geração das métricas de cada link
    # ---------------
    rows = []
    for i, (src, dst) in enumerate(pairs, start=1):

        # Acesso EDGE -> FOG
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
            # Fallback (não deveria acontecer nesta topologia E–F–C)
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


# ## Fases temporais da simulação ##
# A simulação completa se divide em até três fases:
# - pre_disaster
# - during_disaster
# - post_disaster
# A função abaixo é usada em vários módulos para decidir se o timestamp t:
# 1) está antes do desastre,
# 2) dentro do intervalo do desastre,
# ou 3) após o desastre.
# Essa resposta determina como os modelos probabilísticos serão aplicados.

# In[33]:


def phase_from_time(t, cfg):
    if not cfg["disaster"]:
        return "normal"
    if t < cfg["disaster_start"]:
        return "pre_disaster"
    if t < cfg["disaster_end"]:
        return "during_disaster"
    return "post_disaster"


# In[34]:


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


# Nesta parte do notebook, analisaremos a função responsável por gerar a série temporal de métricas de cada enlace da rede (links).
# Essa série temporal é uma das partes mais importantes do dataset, pois modela:
# - latência,
# - jitter,
# - perda,
# - throughput,
# - ocupação de fila,
# - queda do link (is_up),
# - causa da degradação.
# O comportamento dessas métricas muda dependendo do cenário (C1–C5) e dependendo da fase temporal (pre, during, post-disaster).

# ## Objetivo da função generate_link_timeseries() ##
# Essa função:
# - recebe a tabela de links (topologia estática),
# - aplica regras e distribuições estatísticas para cada timestamp,
# - gera efeitos de desastre, flapping, congestionamento etc.,
# - devolve uma tabela longa com uma linha por (link × timestamp).
# A saída possui dezenas, centenas ou milhares de linhas, dependendo dos parâmetros de duração e step.

# In[35]:


def generate_link_timeseries(links, cfg, duration, step):
    """
    Gera séries temporais dos links.

    Correções importantes:
    - latência e jitter nunca ficam negativos (truncamento físico);
    - toda a degradação (desastre, flapping, congestionamento) é aplicada
      diretamente nos links, que serão a base para os fluxos.
    """

    timestamps = np.arange(0, duration, step)
    rows = []

    for _, link in links.iterrows():
        for t in timestamps:

            phase = phase_from_time(t, cfg)

            # Estado base (rede "normal")
            latency = max(0.1, np.random.normal(link["base_latency_ms"], 3))
            jitter  = max(0.1, np.random.normal(2, 1))
            loss    = np.random.uniform(0, link["base_loss_rate"])
            is_up   = 1
            cause   = "none"

            # -------- Fase de desastre --------
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
                        # Queda ocasional de link (C2)
                        if random.random() < 0.1:
                            is_up = 0
                            loss  = 1.0

                # -------- Pós-desastre --------
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

            # -------- Congestionamento (C4) --------
            if cfg["high_traffic"]:
                loss   += np.random.uniform(0.02, 0.12)
                jitter *= np.random.uniform(1.1, 1.5)
                if phase != "pre_disaster":
                    cause = "congestionamento"

            # Normalizações finais
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


# ## Introdução aos próximos geradores: ##
# Nesta etapa, estudaremos três geradores essenciais:
# 1. generate_flows()
# Responsável por criar os fluxos lógicos entre nós da rede, modelados como:
# telemetria crítica,
# controle de atuadores,
# vídeo de drones,
# alertas à população,
# logs em lote,
# best-effort.
# Esses fluxos possuem requisitos diferentes: latência, confiabilidade, prioridade e padrões de tráfego.
# 2. generate_events()
# Simula eventos externos como:
# terremoto,
# aftershock,
# quedas de links,
# flapping.
# Eventos são fundamentais para cenários C2–C5.
# 3. generate_control_actions()
# Simula ações tomadas pelo sistema (fog ou cloud) para restaurar SLA, como:
# reroteamento de fluxo,
# alteração de prioridade,
# redução de carga (throttle),
# migração de serviço.

# In[36]:


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


# ## Explicando o gerador de eventos ##
# Eventos representam ocorrências externas que afetam a rede.
# No ITERATION-D, consideramos:
# - terremoto (impacto global),
# - aftershock (impacto global de menor intensidade),
# - quedas de links individuais,
# - flapping em cenários específicos.
# Esses eventos servem como gatilhos para degradação e ações de controle.

# In[37]:


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


# ## O gerador de ações de controle ##
# A função generate_control_actions() simula o que o sistema tenta fazer para restaurar SLA após o desastre:
# - reroute_flow → mudar caminho
# - promote_flow_priority → aumentar prioridade
# - throttle_video → reduzir tráfego de vídeo
# - migrate_service → mover serviço para outro nó
#   
# Essas ações representam algumas estratégias adaptativas que podem ser implementadas.

# In[38]:


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


# Nesta parte, estudaremos a função mais “alto nível” do simulador: generate_flow_timeseries()
# Ela produz, para cada fluxo da rede:
# - latência fim-a-fim,
# - jitter,
# - bytes entregues,
# - pacotes descartados,
# - cumprimento de SLA,
# - caminho lógico (path_links),
# - fase temporal da simulação.
#   
# Ao contrário da série temporal de links, que trabalha por enlace, esta função trabalha por fluxo, integrando:
# 
# - comportamento da aplicação,
# - requisitos de SLA,
# - impacto do desastre,
# - degradação acumulada,
# - possíveis efeitos de congestionamento ou flapping.

# In[39]:


def generate_flow_timeseries(flows, links, nodes, link_timeseries, cfg, duration, step):
    """
    Gera séries temporais dos fluxos *a partir* dos links no caminho.

    - path_links vem de compute_flow_path (ex.: "L1>L3");
    - em cada timestamp t, buscamos as métricas dos links em link_timeseries;
    - latência fim a fim  = soma das latências dos links;
    - jitter fim a fim    = soma dos jitters dos links;
    - perda fim a fim     = 1 - ∏ (1 - loss_i);
    - throughput_mbps     = min(throughput_mbps dos links)  (gargalo);
    - delivered/dropped   = derivados de throughput + perda.

    A saída inclui tanto throughput em Mbps quanto bytes por intervalo.
    """

    timestamps = np.arange(0, duration, step)

    # Precomputar caminhos de cada fluxo
    flow_paths_str = {}
    flow_paths_list = {}

    for _, flow in flows.iterrows():
        path_str = compute_flow_path(flow, links, nodes)  # ex: "L1>L3"
        flow_paths_str[flow["flow_id"]] = path_str
        if path_str:
            flow_paths_list[flow["flow_id"]] = path_str.split(">")
        else:
            flow_paths_list[flow["flow_id"]] = []

    rows = []

    for _, flow in flows.iterrows():
        flow_id = flow["flow_id"]
        path_links = flow_paths_list[flow_id]

        # Se o fluxo não tem caminho válido, pulamos
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

            # Coleta métricas de cada link do caminho
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

            # Se não achamos nenhum dado de link para esse t, ignoramos
            if not lat_list:
                continue

            # -------------------------
            # Composição fim a fim
            # -------------------------

            # Latência/jitter: soma
            lat_e2e = sum(lat_list)
            jit_e2e = sum(jit_list)

            # Perda composta: 1 - ∏(1 - loss_i)
            success_prob = 1.0
            for li in loss_list:
                success_prob *= (1.0 - li)
            loss_e2e = 1.0 - success_prob
            loss_e2e = min(max(loss_e2e, 0.0), 1.0)

            # Throughput fim a fim (gargalo do caminho)
            if thr_list:
                thr_e2e_mbps = max(0.0, min(thr_list))
            else:
                thr_e2e_mbps = 0.0

            # Bytes entregues/descartados no intervalo
            # throughput (Mbps) -> bytes/s -> bytes/intervalo
            # 1 Mbps = 1e6 bits/s = 1e6/8 bytes/s

            bytes_per_sec = (thr_e2e_mbps * 1e6) / 8.0
            interval_seconds = step

            total_bytes_interval = bytes_per_sec * interval_seconds

            delivered_bytes = total_bytes_interval * (1.0 - loss_e2e)
            dropped_bytes   = total_bytes_interval * loss_e2e

            # Para manter valores "arrumados"
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
        "throughput_mbps",            # throughput E2E no fluxo
        "delivered_bytes_interval",   # bytes efetivamente entregues no intervalo
        "dropped_bytes_interval",     # bytes "perdidos" no intervalo
        "sla_met",
        "phase"
    ])


# ## O papel da run_generator ##
# Até agora, vimos vários blocos da ferramenta:
# - get_scenario_config → define o comportamento global do cenário (C1–C5).
# - generate_nodes → cria a topologia de nós (edge, fog, cloud).
# - generate_links (ou generate_links_realistic) → cria os enlaces com latência, perda e banda base.
# - generate_link_timeseries → gera as métricas de rede por link ao longo do tempo.
# - generate_flows → cria os fluxos lógicos (aplicações).
# - generate_events → introduz terremotos, aftershocks, quedas de link, flapping.
# - generate_control_actions → simula as ações de controle (fog/cloud).
# - generate_flow_timeseries → gera métricas fim-a-fim por fluxo.
# A função run_generator é o “orquestrador principal”:
# Ela chama todas as outras funções na ordem correta.
# Garante reprodutibilidade (via seed).
# Retorna tudo pronto, em um único dicionário de DataFrames — ideal para Jupyter.

# In[40]:


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
    Gera todos os DataFrames do cenário ITERATION-D e retorna em um dicionário.
    """

    random.seed(seed)
    np.random.seed(seed)

    cfg = get_scenario_config(scenario, duration)

    # Topologia
    nodes = generate_nodes(n_edge, n_fog, n_cloud)
    links = generate_links_topology(nodes, n_links)

    # Fluxos (definidos com base na topologia)
    flows = generate_flows(nodes, links, cfg, duration)

    # Séries temporais de links
    link_ts = generate_link_timeseries(links, cfg, duration, step)

    # Eventos e ações
    events  = generate_events(links, cfg, duration)
    actions = generate_control_actions(flows, cfg)

    # Séries temporais de fluxos — agora baseadas em link_ts
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


# Agora que todas as funções internas já foram definidas, vamos finalmente executar o gerador completo usando run_generator().
# 
# No bloco abaixo:
# - Executamos a simulação para um cenário específico (C1–C5).
# - Recebemos os sete DataFrames gerados:
#     - nodes
#     - links
#     - link_timeseries
#     - flows
#     - flow_timeseries
#     - events
#     - control_actions
#       
# Salvamos todos esses DataFrames em arquivos .csv com nomes padronizados, facilitando processamento e análise posterior.
# 

# In[41]:


# Executa o gerador sintético de dados para o cenário escolhido

scenario = "C2"

data = run_generator(
    scenario=scenario,   # Cenário da simulação (C1=estável, C2=desastre, C3=recuperação,
                     # C4=saturação urbana, C5=flapping)

    n_edge=20,       # Número de dispositivos na camada EDGE
    n_fog=5,         # Número de nós intermediários na camada FOG
    n_cloud=1,       # Número de datacenters na camada CLOUD

    n_links=25,      # Número total de enlaces a serem gerados


    duration=3600,   # Duração total da simulação (em segundos)
    step=5,         # Intervalo entre pontos da série temporal (segundos)

    seed=123         # Seed para garantir reprodutibilidade
)


# In[42]:


# Salvamento automático de todos os arquivos gerados
# Cada chave no dicionário 'data' vira um arquivo CSV.

for name, df in data.items():

    # O nome do arquivo segue o padrão:
    #   iterationD_<nome do dataframe>_<cenário>.csv
    filename = f"iterationD_{name}_{scenario}.csv"

    # Exporta o DataFrame para CSV
    df.to_csv(filename, index=False)

    # Log informativo
    print("Arquivo salvo:", filename)


# In[ ]:




