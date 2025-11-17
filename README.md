# ITERATION-D Synthetic Dataset Generator
### Projeto CAPES–STIC–AMSud  
### Universidade Federal de Uberlândia (UFU) • Uruguai • Chile • França

Este repositório contém o **gerador sintético de dados de monitoramento de rede** utilizado no projeto **ITERATION-D – CAPES–STIC–AMSud**.  
O objetivo é simular **arquiteturas distribuídas de Edge–Fog–Cloud** sob diferentes condições operacionais e cenários de desastre, produzindo um conjunto abrangente de dados para análise, detecção de falhas, resiliência e tomada de decisão distribuída.

---

## Objetivos

O gerador produz **datasets realistas** para avaliação de:

- comportamento de redes distribuídas (edge/fog/cloud);
- latência, jitter, perda e throughput em nível de link;
- desempenho de fluxos ponta-a-ponta (e2e);
- impacto de desastres naturais no tráfego de rede;
- mecanismos de controle e mitigação executados por Fog/Cloud;
- reconstrução de rotas e path inference;
- estratégias de resiliência e SLA preservation.

---

## Arquitetura Simulada

A ferramenta modela uma rede com três camadas:

1. **Edge**  
   - Dispositivos com baixa capacidade computacional  
   - Três tipos possíveis:  
     - `sensor` → telemetria crítica  
     - `UAV` → fluxo de vídeo  
     - `wearable` → tráfego best-effort  

2. **Fog**  
   - Gateways distribuídos  
   - Ponto intermediário entre Edge e Cloud  

3. **Cloud**  
   - Datacenters regionais  
   - Processamento de tráfego crítico ou agregado  

A **topologia é sempre coerente**:
- cada Edge → exatamente 1 Fog  
- cada Fog → pelo menos 1 Cloud  
- links extras opcionais respeitam apenas Edge→Fog e Fog→Cloud  

---

## Cenários Disponíveis

| Cenário | Descrição |
|--------|-----------|
| **C1** | Rede estável, sem desastre |
| **C2** | Desastre padrão (latência↑, perda↑, quedas moderadas) |
| **C3** | Desastre + recuperação adaptativa |
| **C4** | Desastre + congestionamento urbano |
| **C5** | Desastre + flapping / instabilidade severa |

Cada cenário afeta **link_timeseries** e, por consequência, **flow_timeseries**.

---

## Arquivos gerados (CSV)

Ao executar o gerador, os seguintes arquivos são criados:

### **1. nodes.csv**
Lista de nós com atributos estruturais:
- node_id  
- layer (edge/fog/cloud)  
- device_type  
- region  
- compute_capacity  
- is_critical  

### **2. links.csv**
Grafo físico da rede:
- link_id  
- src_node_id  
- dst_node_id  
- tecnologia (LTE, fibra, etc.)  
- latência base  
- banda base  
- perda base  

### **3. link_timeseries.csv**
Séries temporais por link:
- timestamp  
- latency_ms  
- jitter_ms  
- loss_rate  
- throughput_mbps  
- queue_occupancy  
- is_up  
- causa da degradação  

### **4. flows.csv**
Fluxos ponta-a-ponta:
- FLx  
- origem → destino  
- tipo de aplicação  
- requisitos de SLA  

### **5. flow_timeseries.csv**
Séries temporais dos fluxos:
- latência e jitter **e2e (soma dos links)**  
- perda composta  
- throughput **(gargalo do caminho)**  
- delivered / dropped bytes  
- fase do cenário (normal, desastre, recuperação)  

### **6. events.csv**
Eventos de desastre:
- terremoto, aftershock, queda de links, flapping  

### **7. control_actions.csv**
Ações tomadas pelos controladores Fog/Cloud:
- reroute  
- promote priority  
- throttle  
- migrate  

---

## Como executar (arquivo Python)

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

# Salvar CSVs
for name, df in data.items():
    df.to_csv(f"{name}.csv", index=False)
