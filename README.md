# CASSCO Model Simulations

This repository contains simulation code for the **CASSCO Model** (Complex Adaptive System of Science Communication), a framework for analyzing strategic interactions in science communication using game theory and complex adaptive systems principles.

## Overview

The CASSCO model conceptualizes science communication as a complex adaptive system where different actors (research institutions and researchers) select strategies based on their roles (altruistic or self-interested) and preferred communication modes (knowledge dissemination, dialogue, or participation). The model simulates:

- Strategic decision-making by actors in science communication
- Bayesian learning and belief updates based on observed outcomes
- Dynamic evolution of exigence (communication need) over time
- Equilibrium points for different parameter configurations

Two scenarios are implemented: **Generative AI** and **Citizen Science**, each with different impact assessments reflecting their unique contexts.

## Key Features

- **Game-theoretic modeling** of strategic interactions between organizations and researchers
- **Bayesian learning mechanisms** simulating belief updates
- **Exigence dynamics** showing how communication needs evolve over time
- **Parameter sensitivity analysis** to understand effects of learning rates and thresholds
- **Excel exports** of all calculation steps for transparency and validation
- **Publication-quality visualizations** for direct use in academic papers

## Visualization Examples

The code generates three main figures:

1. **Scenario Comparison** - Impact matrices and strategy evolution for both scenarios
2. **Model Dynamics** - Utility evolution, exigence dynamics, and equilibrium analysis
3. **Strategic Aspects** - Alpha-strategy relationships and parameter effects

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cassco-simulations.git
cd cassco-simulations

# Install required packages
pip install numpy matplotlib pandas openpyxl
```

## Usage

Run the main simulation code:

```bash
python cassco-model-updated.py
```

This will:
1. Create a `simulations/` directory if it doesn't exist
2. Run all simulations for both scenarios
3. Generate PDF visualizations in the `simulations/` folder
4. Export detailed calculation Excel files for each simulation step

## Customization

You can modify key parameters in the `ModelParams` class:
- `alpha_O` and `alpha_R`: Weight between impact and private benefit (0-1)
- `beta`: Learning rate for exigence updates
- `theta`: Threshold impact value
- `scenario`: 'gen_ai' or 'citizen_science'

To add randomness to impact values, modify the observed impact calculation in the `calculate_game_step()` function.

## File Structure

```
cassco-simulations/
├── CASSCO-simulations.py       # Main simulation code
├── simulations/                  # Output directory
│   ├── figure1_scenarios_comparison.pdf
│   ├── figure2_model_evolution_equilibrium.pdf
│   ├── figure3_strategic_aspects.pdf
│   ├── GenerativeAI_Calculations_Step0.xlsx
│   ├── CitizenScience_Calculations_Step0.xlsx
│   └── ...                       # Additional simulation results
└── README.md                     # This file
```

## Theoretical Background

The CASSCO model is based on the integration of complex adaptive systems theory and game theory applied to science communication dynamics. It models:

1. **Actor Decision Making**: Organizations and researchers choose strategies to maximize expected utility
2. **Belief Updates**: Bayesian updating of beliefs about other actors' strategies
3. **Exigence Dynamics**: How communication needs evolve based on impact and parameters

For the theoretical foundation, see the related paper (citation below).


## License

This project is licensed under the MIT License - see the LICENSE file for details.

