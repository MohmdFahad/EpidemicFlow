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

----------------> E P I D E M I C S I M U L A T I O N R E S U L T S <---------------

Epidemic Duration: 79 days\
Peak Active Infections: 253 (on Day 28)\
Total Infected: 1112\
Total Recovered: 1112\
Never Infected: 111

--- Behavior and Awareness ---

Individuals who became aware: 826\
Average Awareness Efficacy: 23.6\
Individuals who quarantined: 469

--- Recovery Time Statistics ---

Mean Recovery Duration: 7.48 days\
Recovery Std Dev: 1.97 days\
Min Recovery Duration: 3 days\
Max Recovery Duration: 15 days

---

## ▶️ How to Run

Clone the repository and run the main simulation module:

```bash
git clone https://github.com/MohmdFahad/EpidemicFlow.git
cd EpidemicFlow
python -m src.epidemicflow
```

You’ll then be prompted to input the simulation parameters (infection probability, awareness efficacy, grid size, etc.).
All results will be logged automatically in /data/results/ as a .csv file and printed to the console.

---

## 🧱 Repository Structure

EpidemicFlow\
└── src # Core simulation code\
│ └─ epidemicflow.py # Entry point\
│ └─ simulation.py # Daily simulation loop\
│ └─ infection.py # Infection and recovery dynamics\
│ └─ awareness.py # Awareness and behavioral modeling\
│ └─ grid_utils.py # Grid setup and neighbor functions\
│\
└─ data # Simulation input/output\
│ └─ results\
│ └─ initial_conditions\
│
└─ examples # Example simulation runners
│ └─ example_scenario1.py\
│ └─ example_scenario2.py\
│ └─ example_scenario3.py\
│ └─ results # results from example runs\
│ └─ README.md # Brief usage instructions for examples\ 
│\
└─ docs # Documentation (model, parameters, results)\
│ └─ model_description.md # Explains math, transitions, and logic\
│ └─ results_summary.md # Scenario results, plots, interpretation\
│\
└─ README.md # Project overview, usage, and model explanation\
└─ requirements.txt # pip dependencies (numpy, pandas)\
└─ environment.yml # Conda environment file
└─ .gitignore # Ignores venv, cache, and results

---

## 🧰 Requirements

- Python 3.10+
- NumPy
- Pandas

Install dependencies with:

## Option 1: Using conda (recommended)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate epidemicflow
```

## Option 2: Using pip

```bash
pip install -r requirements.txt
```

---

## 🧠 Applications

EpidemicFlow can be used for:
- Studying behavioral responses to disease spread.
- Evaluating the effect of awareness campaigns on outbreak size.
- Teaching epidemiological modeling and probabilistic simulations.
- Comparing infection control measures such as quarantine vs. awareness.

---

## 🧾 License


This project is licensed under the **Creative Commons Attribution–NonCommercial 4.0 International License (CC BY-NC 4.0)**.

You are free to:
- Share — copy and redistribute the material in any medium or format  
- Adapt — remix, transform, and build upon the material  

Under the following terms:
- Attribution — you must give appropriate credit to the author.  
- NonCommercial — you may not use the material for commercial purposes.

Read the full licence text here: [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)

---

## 👤 Author

**Mohamed Fahad**  
Copyright © 2025  
Licensed under the [CC BY-NC 4.0 License](./LICENSE)




