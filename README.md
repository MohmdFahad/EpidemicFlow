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

| Label | Meaning | Transition Triggers |
|--------|----------|--------------------|
| `0` | Susceptible | Can become infected or aware |
| `1` | Infected | Infects neighbors; recovers after γ-distributed days |
| `4` | Aware | Reduced infection risk based on efficacy |
| `5` | Quarantined | Limited contact; low infection risk |
| `2` | Recovered | Immune thereafter |

---

### ⚙️ Core Mathematical Components

#### 1. Infection Cycle
The infection update frequency (`n_days`) follows a logistic decay:
\[
n = n_{min} + \frac{n_{max} - n_{min}}{1 + e^{a \cdot (I - p_0)}}
\]
- **\( I \)** = infection fraction  
- **\( p_0 \)** = pivot where slowdown begins  
- **\( a \)** = steepness of decay  

#### 2. Awareness Cycle
Awareness updates follow a similar adaptive curve:
\[
n_{aware} = n_{min} + \frac{n_{max} - n_{min}}{1 + e^{-a \cdot (A - p_0)}}
\]
- **\( A \)** = fraction of aware individuals  
- Inverse mapping → awareness spreads faster early, slower when most are aware.  

#### 3. Awareness Adoption Probability
Each susceptible individual calculates awareness likelihood using:
\[
P_{aware} = P_{base} + (1 - P_{base}) \cdot \sigma(k(x - b))
\]
where  
- \( \sigma(z) = \frac{1}{1 + e^{-z}} \) is the sigmoid function  
- \( x = \text{aware neighbor fraction} + \alpha \cdot \text{infection\_percent} \)  
- \( k, b \) = dynamic steepness and threshold parameters  

#### 4. Recovery Time
Recovery time follows a **Gamma Distribution**:
\[
\text{days} \sim \Gamma(k, \theta)
\]
where  
\( k = \frac{\text{mean}^2}{\text{variance}} \), \( \theta = \frac{\text{variance}}{\text{mean}} \)  
ensuring realistic variation — most recover near the mean, with few long recoveries.

---

## 📊 Example Simulation Output

