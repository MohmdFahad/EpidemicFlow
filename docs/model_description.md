# 🧠 EpidemicFlow — Model Description

## 1. Overview

**EpidemicFlow** simulates epidemic spread across a grid-based population, integrating behavioral responses such as awareness and quarantine.  
The goal is to explore how infection probability, awareness, and behavioral feedback loops shape epidemic dynamics in a controlled computational environment.

Each cell in the grid represents one individual with an evolving health and behavioral state.

---

## 2. Individual States

| Label | State | Description |
|-------|--------|-------------|
| **0** | Susceptible | Healthy individual who has not yet been infected or influenced by awareness. |
| **1** | Infected | Currently infected and capable of transmitting the disease. |
| **2** | Recovered | Has recovered and is immune for the remainder of the simulation. |
| **3** | Susceptible and Quarantined | Healthy but self-isolating due to nearby infections. |
| **4** | Susceptible and Aware | Healthy but aware of the disease and practicing preventive behavior, lowering infection risk. |
| **5** | Infected and Aware | Infected individual who is aware and less likely to infect others. |
| **6** | Infected and Quarantined | Infected individual in quarantine, strongly reducing further transmission. |

---

## 3. Core Model Mechanics

### 3.1 Grid and Neighbourhood
- The population is represented as a **2D NumPy array** of shape `(N, N)`.
- Each individual interacts with its **eight neighbours** — the four orthogonal (up, down, left, right) and the four diagonal corners (up-left, up-right, down-left, down-right).
- These local interactions drive both infection and awareness spread.

---

### 3.2 Infection Dynamics

Infection spreads based on **local contact probability** and adaptive infection frequency.

n_inf = n_min + (n_max - n_min) / (1 + exp(a * (I - p0)))


- **I** — infection fraction (global)
- **p₀** — pivot: infection level where spread begins slowing
- **a** — steepness of the curve
- **n_inf** — number of days between infection spread events

This allows infection to start rapidly and slow as population saturation increases.

---

### 3.3 Awareness Dynamics

Awareness spreads both **locally (through neighbours)** and **globally (through general information flow)**.

n_aware = n_min + (n_max - n_min) / (1 + exp(-a * (A - p0)))


- **A** — fraction of aware individuals  
- Awareness spreads faster early on, but slows as saturation increases.

Each susceptible person calculates their **probability of becoming aware**:

P_aware = P_base + (1 - P_base) * sigmoid(k * (x - b))


where:
- **sigmoid(z)** = 1 / (1 + exp(-z))
- **x** = aware_neighbour_fraction + α * infection_percent
- **k** — steepness  
- **b** — threshold  

This models **awareness adoption** as a smooth behavioral response.

---

### 3.4 Quarantine Mechanism
- Individuals surrounded by infections may enter quarantine with a given probability.  
- Quarantined individuals reduce their **infection probability** significantly (to simulate limited contact).  
- Quarantine can apply to both susceptible and infected individuals.

---

### 3.5 Recovery Time (Gamma Distribution)

Recovery times are drawn from a **Gamma distribution**, introducing realistic variability:

days ~ Gamma(k, θ)

with:
- k = (mean²) / variance
- θ = variance / mean


- Most recover around the mean duration.  
- A few recover much earlier or later, creating a natural tail in the recovery curve.

---

## 4. Simulation Metrics

At the end of each simulation, the following statistics are logged:
- **Epidemic Duration:** total days until infection ceases.
- **Peak Active Infections:** max simultaneous infected count and day of peak.
- **Total Infected:** number of individuals ever infected.
- **Never Infected:** individuals who remained susceptible throughout.
- **Individuals who became aware**
- **Average Awareness Efficacy**
- **Individuals who quarantined**
- **Recovery Stats:** mean, standard deviation, min, and max.

---

## 5. Example Trends and Interpretations

Typical observations:
- Increasing awareness efficacy lowers infection peak but lengthens epidemic duration.
- High quarantine adherence produces smaller but longer epidemics.
- Higher infection probability accelerates peaks and reduces the effect of awareness spread.
- Gamma-distributed recovery introduces visible stochastic recovery patterns.

---

## 6. Limitations & Future Work
- Currently uses a **2D grid** (no long-range transmission).  
- No stochastic mutation or immunity loss.  
- Future versions could include:
  - Multi-strain dynamics  
  - Time-dependent awareness decay  
  - Demographic segmentation (age or mobility)
