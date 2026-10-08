# 🧠 EpidemicFlow — Model Description

## 1. Overview

**EpidemicFlow** simulates epidemic spread across a grid-based population, integrating behavioral responses such as awareness and quarantine.  
The goal is to explore how infection probability, awareness, and behavioral feedback loops shape epidemic dynamics in a controlled computational environment.

Each cell in the grid represents one individual with an evolving health and behavioral state.

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

## 3. Core Model Mechanics

### 3.1 Grid and Neighbourhood
- The population is represented as a **2D NumPy array** of shape `(N, N)`.
- Each individual interacts with its **eight neighbours** — the four orthogonal (up, down, left, right) and the four diagonal corners (up-left, up-right, down-left, down-right).
- These local interactions drive both infection and awareness spread.

### 3.2 Infection Dynamics

Infection spreads based on **local contact probability** and adaptive infection frequency.

n_inf = n_min + (n_max - n_min) / (1 + exp(a * (I - p0)))


- **I** — fraction of the population currently infected
- **p₀** — pivot: infection level where spread begins slowing
- **a** — steepness of the curve
- **n_inf** — number of days between infection spread events

This allows infection to start rapidly and slow as population saturation increases.

### 3.3 Awareness Dynamics

Awareness spreads in events every *n* days:

n_aware = clip( n_max · exp(−α·I + β·A), n_min, n_max )

- **I** — fraction of the population currently infected
- **A** — fraction currently aware
- **α, β, n_min, n_max** — hand-tuned per parameter regime (`awareness.py`)

On each awareness event:

1. **Global campaign (once per event).** Each unaware susceptible person becomes aware with a small probability. The probability is between 0.2% and 1%, rises with infection level and falls as awareness saturates.
2. **Local spread.** Each unaware neighbour of an aware person becomes aware with probability `awareness_rate · sigmoid(k·(x − b))`, where `x = aware_neighbour_share + 1.5·I`. Both the steepness `k` and the threshold `b` depend on the share of aware neighbours.

When an unaware person is exposed to infection (and is not surrounded by infection), they may adopt protective behaviour in time instead of being infected:

P_aware = awareness_rate · ( P_base + (1 − P_base) · sigmoid(k·(x − b)) )

with `x = 0.6·aware_neighbour_share + 0.4·infected_neighbour_share`, `k = 6`, `b = 0.45 − 0.15·I` and `P_base = 0.01`.

### 3.4 Quarantine Mechanism
- An exposed susceptible person with **at least half** their neighbours infected quarantines with probability `quarantine_chance` instead of being infected.
- An infected person with at least half their neighbours infected quarantines with the same probability, and **stops spreading**.
- Quarantine lasts 7–14 days (uniform) and counts down once per day. Afterwards the person is aware (state 3 → 4, 6 → 5).
- A quarantined susceptible person who is exposed has a **90% lower** chance of infection. If infected, they stay in quarantine (state 6).

### 3.5 Recovery Time (Gamma Distribution)

Recovery times are drawn from a **Gamma distribution**, introducing realistic variability:

days ~ Gamma(k, θ)

with:
- k = (mean²) / variance
- θ = variance / mean


- Most recover around the mean duration.  
- A few recover much earlier or later, creating a natural tail in the recovery curve.

## 4. Simulation Metrics

At the end of each simulation, the following statistics are logged:
- **Epidemic Duration:** days until nobody is infected (including aware and quarantined infected).
- **Peak Active Infections:** max simultaneous infected count and day of peak.
- **Total Infected / attack rate:** number (and share) of individuals ever infected.
- **Never Infected:** individuals who remained susceptible throughout.
- **Individuals who became aware**
- **Individuals who quarantined**
- **Recovery Stats:** mean, standard deviation, min, and max.

## 5. Example Trends and Interpretations

Typical observations:
- Increasing awareness efficacy lowers infection peak but lengthens epidemic duration.
- High quarantine adherence produces smaller but longer epidemics.
- Higher infection probability accelerates peaks and reduces the effect of awareness spread.
- Gamma-distributed recovery introduces visible stochastic recovery patterns.

## 6. Limitations & Future Work
- Currently uses a **2D grid** (no long-range transmission).  
- No stochastic mutation or immunity loss.  
- Future versions could include:
  - Multi-strain dynamics  
  - Time-dependent awareness decay  
  - Demographic segmentation (age or mobility)
