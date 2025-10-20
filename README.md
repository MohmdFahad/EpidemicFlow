# 🦠 EpidemicFlow: Behavioral Epidemic Simulation in Python

**EpidemicFlow** is a grid-based epidemic simulation framework that integrates **infection dynamics**, **behavioral awareness**, and **quarantine mechanisms** to study how individual behavior affects disease spread.  
Developed entirely in **Python (NumPy & Pandas)**, it models how awareness, infection probability, and recovery variability interact to produce realistic epidemic trends.

---

## 🚀 Features

- 🧬 **Grid-based population model:** each cell represents an individual with a state (Susceptible, Infected, Aware, Quarantined, Recovered).  
- 📈 **Probabilistic infection spread:** neighbor-based infection propagation using tuned logistic cycles.  
- 🧠 **Behavioral modeling:** awareness spread follows a **sigmoid adoption probability**, dynamically influenced by infection levels and awareness saturation.  
- 🧮 **Gamma-distributed recovery:** realistic variability in recovery time (shape and scale derived from user input).  
- 🔄 **Awareness & infection cycles:** adaptive logistic timing mechanisms control how frequently infection and awareness spread.  
- 📊 **Automatic metric tracking:** uses Pandas to log daily infection counts, awareness levels, and recovery statistics.  

---

## 🧩 Model Overview

Each simulation step represents one **day**.  
Individuals can transition between states based on probabilistic and behavioral conditions:



| Label | State | Description |
|-------|--------|-------------|
| **0** | Susceptible | Healthy individual who has not yet been infected or influenced by awareness. |
| **1** | Infected | Currently infected and capable of transmitting the disease. |
| **2** | Recovered | Has recovered and is immune for the remainder of the simulation. |
| **3** | Susceptible and Quarantined | Healthy but self-isolating due to nearby infections; temporarily removed from exposure. |
| **4** | Susceptible and Aware | Healthy but aware of the disease and practicing preventive behavior, lowering infection risk. |
| **5** | Infected and Aware | Infected individual who is aware and thus less likely to infect others. |
| **6** | Infected and Quarantined | Infected individual under quarantine, significantly reducing further transmission. |

---

### ⚙️ Core Mathematical Components

#### 1. Infection Cycle
The infection update frequency (`n_days`) follows a logistic decay:

**n** = n_min + (n_max - n_min) / (1 + exp(a * (I - p0)))

- **\( I \)** = infection fraction  
- **\( p_0 \)** = pivot where slowdown begins  
- **\( a \)** = steepness of decay  

#### 2. Awareness Cycle
Awareness updates follow a similar adaptive curve:

**n** = n_min + (n_max - n_min) / (1 + exp(a * (I - p0)))

- **\( A \)** = fraction of aware individuals  
- Inverse mapping → awareness spreads faster early, slower when most are aware.  

#### 3. Awareness Adoption Probability
Each susceptible individual calculates awareness likelihood using:

**P_aware** = P_base + (1 - P_base) * sigmoid(k * (x - b))

where  
- **sigmoid(z)** = 1 / (1 + exp(-z))  
- **x** = aware_neighbor_fraction + α * infection_percent
- **k** and **b** → dynamic steepness and threshold parameters

#### 4. Recovery Time
Recovery time follows a **Gamma Distribution**:

days ~ **Gamma(k, θ)**
where:
-    **k** = (mean^2) / variance
-    **θ** = variance / mean


---

## 📊 Example Simulation Output

