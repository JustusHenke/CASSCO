import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec



class OrigModelParams:
    def __init__(self, scenario='gen_ai'):
        # Common parameters for both scenarios
        self.P_altruistic = 0.3  # Private benefit for altruistic strategies
        self.P_egoistic = 0.7    # Private benefit for egoistic strategies
        
        self.num_agents = 2  # For two-agent model

        # Set the scenario
        self.scenario = scenario
        
        # Apply scenario-specific parameters
        self.set_scenario_params(scenario)
    
    def set_scenario_params(self, scenario):
        """Set all parameters based on scenario"""
        if scenario == 'gen_ai':
            # Generative AI scenario parameters
            self.M_O = 0.5       # Max impact for organization
            self.M_R = 0.5       # Max impact for researcher
            self.alpha_O = 0.5   # Weight for organization (impact vs private benefit)
            self.alpha_R = 0.5   # Weight for researcher
            self.X_init = 0.5    # Initial exigence
            self.beta = 0.4      # Learning rate
            self.theta = 0.3     # Threshold impact
        elif scenario == 'citizen_science':
            # Citizen Science scenario parameters
            self.M_O = 0.4       # Max impact for organization
            self.M_R = 0.6       # Max impact for researcher
            self.alpha_O = 0.6   # Weight for organization (impact vs private benefit)
            self.alpha_R = 0.7   # Weight for researcher
            self.X_init = 0.2    # Initial exigence
            self.beta = 0.3      # Learning rate
            self.theta = 0.5     # Threshold impact
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

# Model parameters class with scenario-specific settings for three agents
class ModelParams:
    def __init__(self, scenario='citizen_science_museum'):
        # Common parameters
        self.P_altruistic = 0.3  # Private benefit for altruistic strategies
        self.P_egoistic = 0.7    # Private benefit for egoistic strategies


        self.num_agents = 3  # For three-agent model
        
        # Set the scenario
        self.scenario = scenario
        
        # Apply scenario parameters for three-agent model
        self.set_scenario_params(scenario)
    
    def set_scenario_params(self, scenario):
        """Set all parameters based on scenario"""
        if scenario == 'citizen_science_museum':
            # Three-agent Citizen Science scenario with Museum
            self.M_O = 0.3       # Max impact for organization (reduced from 0.4)
            self.M_R = 0.4       # Max impact for researcher (reduced from 0.6)
            self.M_M = 0.3       # Max impact for museum (new actor)
            
            self.alpha_O = 0.6   # Weight for organization (impact vs private benefit)
            self.alpha_R = 0.7   # Weight for researcher
            self.alpha_M = 0.4   # Weight for museum (higher private interest)
            
            self.X_init = 0.2    # Initial exigence
            self.beta = 0.3      # Learning rate
            self.theta = 0.5     # Threshold impact
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

# Define the possible strategies
def get_strategies():
    roles = ['a', 'e']  # a:altruistic, e:egoistic
    modes = ['k', 'd', 'p']  # k:knowledge, d:dialogue, p:participation
    return [(r, m) for r in roles for m in modes]


# Get impact values based on scenario
def get_impact_values(scenario):
    if scenario == 'gen_ai':
        # Impact assessments for the generative AI scenario (from Table 5 in the paper)
        impact_O = {
            ('a', 'k'): 0.7,  # Detailed explanations of AI methods
            ('a', 'd'): 0.5,  # Expert panels on AI validation
            ('a', 'p'): 0.4,  # Public AI validation workshops
            ('e', 'k'): 0.6,  # Presentation of AI validation protocols
            ('e', 'd'): 0.4,  # AI ethics committees
            ('e', 'p'): 0.3   # AI hackathons for validation
        }
        
        impact_R = {
            ('a', 'k'): 0.9,  # Scientific publications on AI validation methods
            ('a', 'd'): 0.7,  # Peer-review discussions on AI methods
            ('a', 'p'): 0.5,  # Open-source validation tools
            ('e', 'k'): 0.8,  # Longitudinal studies on AI's effect
            ('e', 'd'): 0.4,  # AI-focused conference contributions
            ('e', 'p'): 0.3   # Crowdsourcing of AI training
        }
    elif scenario == 'citizen_science':
        # Impact assessments for the citizen science scenario (from Table 4 in the paper)
        impact_O = {
            ('a', 'k'): 0.5,  # Providing resources and expert knowledge
            ('a', 'd'): 0.7,  # Organizing discussion forums and workshops
            ('a', 'p'): 0.9,  # Providing infrastructure for citizen projects
            ('e', 'k'): 0.3,  # Dissemination of research results
            ('e', 'd'): 0.5,  # Networking events with stakeholders
            ('e', 'p'): 0.6   # Large-scale citizen science campaigns
        }
        
        impact_R = {
            ('a', 'k'): 0.5,  # Creation of educational materials
            ('a', 'd'): 0.7,  # Conducting citizen dialogues
            ('a', 'p'): 0.9,  # Joint data collection and analysis with citizens
            ('e', 'k'): 0.3,  # Writing academic articles
            ('e', 'd'): 0.5,  # Presentations at conferences
            ('e', 'p'): 0.6   # Leading citizen science projects
        }
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return impact_O, impact_R

# Get impact values for three agents in the museum scenario
def get_impact_values_three_agents(scenario):
    if scenario == 'citizen_science_museum':
        # Impact assessments for the three-agent citizen science scenario
        
        # Organization impact (universities) - less participation opportunities, more dissemination/dialogue
        impact_O = {
            ('a', 'k'): 0.5,  # Providing resources and expert knowledge
            ('a', 'd'): 0.7,  # Organizing discussion forums and workshops
            ('a', 'p'): 0.9,  # Providing infrastructure for citizen projects
            ('e', 'k'): 0.3,  # Dissemination of research results
            ('e', 'd'): 0.5,  # Networking events with stakeholders
            ('e', 'p'): 0.6   # Large-scale citizen science campaigns
        }
        
        impact_R = {
            ('a', 'k'): 0.5,  # Creation of educational materials
            ('a', 'd'): 0.7,  # Conducting citizen dialogues
            ('a', 'p'): 0.9,  # Joint data collection and analysis with citizens
            ('e', 'k'): 0.3,  # Writing academic articles
            ('e', 'd'): 0.5,  # Presentations at conferences
            ('e', 'p'): 0.6   # Leading citizen science projects
        }
        
        # Museum impact (new actor with high impact in participation)
        impact_M = {
            ('a', 'k'): 0.6,  # Providing educational materials and exhibits
            ('a', 'd'): 0.7,  # Hosting science discussions and dialogues
            ('a', 'p'): 0.9,  # Running citizen science programs at museum
            ('e', 'k'): 0.5,  # Creating popular science content highlighting museum
            ('e', 'd'): 0.6,  # Organizing media events for visibility
            ('e', 'p'): 0.8   # Branded citizen science projects to attract visitors
        }
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return impact_O, impact_R, impact_M

# Calculate Bayesian likelihood for belief update (modified for three agents)
def calculate_likelihood(strategy, observed_impact, normalized_impact, strategies):
    """Calculate likelihood based on strategy contribution to observed impact"""
    strategy_impact = normalized_impact[strategy]
    total_impact = sum(normalized_impact.values())
    
    contribution = strategy_impact / total_impact if total_impact > 0 else 0
    likelihood = contribution * observed_impact
    likelihood = max(0.001, likelihood)  # Avoid zero likelihood
    
    return likelihood

# Calculate one step of the game theoretic model
def calculate_game_step(params, beliefs_O=None, beliefs_R=None, t=0, export_excel=False, prev_beliefs_O=None, prev_beliefs_R=None):
    """
    Calculates one step of the game theoretic model following the paper's formulation
    """
    strategies = get_strategies()
    
    # Get impact values for the selected scenario
    impact_O, impact_R = get_impact_values(params.scenario)
    
    # Normalize impacts according to equation 4
    normalized_impact_O = {k: v * params.M_O for k, v in impact_O.items()}
    normalized_impact_R = {k: v * params.M_R for k, v in impact_R.items()}
    
    # Initialize beliefs based on equation 5 if not provided
    if beliefs_O is None or beliefs_R is None:
        sum_impact_O = sum(normalized_impact_O.values())
        sum_impact_R = sum(normalized_impact_R.values())
        
        beliefs_O = {s: normalized_impact_R[s]/sum_impact_R for s in strategies}
        beliefs_R = {s: normalized_impact_O[s]/sum_impact_O for s in strategies}
    
    # Calculate expected impacts according to equation 6
    expected_impacts_O = {}
    expected_impacts_R = {}
    
    for s_O in strategies:
        # Expected impact of O's strategy
        own_impact = normalized_impact_O[s_O]
        other_expected_impact = sum(beliefs_O[s_R] * normalized_impact_R[s_R] for s_R in strategies)
        expected_impacts_O[s_O] = own_impact + other_expected_impact
    
    for s_R in strategies:
        # Expected impact of R's strategy
        own_impact = normalized_impact_R[s_R]
        other_expected_impact = sum(beliefs_R[s_O] * normalized_impact_O[s_O] for s_O in strategies)
        expected_impacts_R[s_R] = own_impact + other_expected_impact
    
    # Calculate expected utilities according to equation 7
    expected_utilities_O = {}
    expected_utilities_R = {}
    
    for s_O in strategies:
        private_benefit = params.P_egoistic if s_O[0] == 'e' else params.P_altruistic
        expected_utilities_O[s_O] = params.alpha_O * expected_impacts_O[s_O] + (1 - params.alpha_O) * private_benefit
    
    for s_R in strategies:
        private_benefit = params.P_egoistic if s_R[0] == 'e' else params.P_altruistic
        expected_utilities_R[s_R] = params.alpha_R * expected_impacts_R[s_R] + (1 - params.alpha_R) * private_benefit
    
    # Choose optimal strategies according to equation 8
    s_O_opt = max(expected_utilities_O, key=expected_utilities_O.get)
    s_R_opt = max(expected_utilities_R, key=expected_utilities_R.get)
    
    # Calculate observed impact based on optimal strategies with random variation
    import random
    
    # Apply randomization similar to Excel formula: MIN(MAX(base_value + RAND()*0.2-0.1, 0), 1)
    observed_impact_O = min(max(normalized_impact_O[s_O_opt] + random.random() * 0.1 - 0.05, 0), 1)
    observed_impact_R = min(max(normalized_impact_R[s_R_opt] + random.random() * 0.1 - 0.05, 0), 1)
    
    # Ensure impacts don't exceed maximum possible values
    observed_impact_O = min(observed_impact_O, params.M_O)
    observed_impact_R = min(observed_impact_R, params.M_R)
    
    # Calculate total observed impact
    observed_total_impact = observed_impact_O + observed_impact_R
    
    # Update beliefs according to equation 9 (Bayesian update)
    # Calculate likelihoods for each strategy
    likelihood_O = {}
    likelihood_R = {}
    
    for s in strategies:
        # Calculate likelihood P(I*|s) for Organization's beliefs about Researcher strategies
        likelihood_O[s] = calculate_likelihood(s, observed_total_impact, normalized_impact_R, strategies)
        
        # Calculate likelihood P(I*|s) for Researcher's beliefs about Organization strategies
        likelihood_R[s] = calculate_likelihood(s, observed_total_impact, normalized_impact_O, strategies)
    
    # Calculate unnormalized posterior beliefs
    unnormalized_posterior_O = {s: likelihood_O[s] * beliefs_O[s] for s in strategies}
    unnormalized_posterior_R = {s: likelihood_R[s] * beliefs_R[s] for s in strategies}
    
    # Calculate normalization factors
    normalization_factor_O = sum(unnormalized_posterior_O.values())
    normalization_factor_R = sum(unnormalized_posterior_R.values())
    
    # Calculate updated beliefs (normalized posterior)
    updated_beliefs_O = {s: v/normalization_factor_O for s, v in unnormalized_posterior_O.items()}
    updated_beliefs_R = {s: v/normalization_factor_R for s, v in unnormalized_posterior_R.items()}
    
    
    # Recalculate expected impacts with updated beliefs
    recalculated_impacts_O = {}
    recalculated_impacts_R = {}
    
    for s_O in strategies:
        # Expected impact of O's strategy with updated beliefs
        own_impact = normalized_impact_O[s_O]
        other_expected_impact = sum(updated_beliefs_O[s_R] * normalized_impact_R[s_R] for s_R in strategies)
        recalculated_impacts_O[s_O] = own_impact + other_expected_impact
    
    for s_R in strategies:
        # Expected impact of R's strategy with updated beliefs
        own_impact = normalized_impact_R[s_R]
        other_expected_impact = sum(updated_beliefs_R[s_O] * normalized_impact_O[s_O] for s_O in strategies)
        recalculated_impacts_R[s_R] = own_impact + other_expected_impact
    
    # Recalculate expected utilities with updated beliefs
    recalculated_utilities_O = {}
    recalculated_utilities_R = {}
    
    for s_O in strategies:
        private_benefit = params.P_egoistic if s_O[0] == 'e' else params.P_altruistic
        recalculated_utilities_O[s_O] = params.alpha_O * recalculated_impacts_O[s_O] + (1 - params.alpha_O) * private_benefit
    
    for s_R in strategies:
        private_benefit = params.P_egoistic if s_R[0] == 'e' else params.P_altruistic
        recalculated_utilities_R[s_R] = params.alpha_R * recalculated_impacts_R[s_R] + (1 - params.alpha_R) * private_benefit
    
    
    # Update exigence according to equation 10
    # To better reflect that more actors can address larger portions of the population,
    # a simple modification to the exigence update function that adjusts the threshold based on the number of actors.
    # scaled_theta = params.theta * (2/params.num_agents)
    normalized_impact = observed_total_impact / (params.num_agents/2)

    # X_next = params.X_init + params.beta * (scaled_theta - observed_total_impact) * params.X_init * (1 - params.X_init)
    X_next = params.X_init + params.beta * (params.theta - normalized_impact) * params.X_init * (1 - params.X_init)
    exigence_change = X_next - params.X_init
    
    
    return {
        'step': t,
        'X_t': params.X_init,
        'X_next': X_next,
        'strategy_O': s_O_opt,
        'strategy_R': s_R_opt,
        'observed_impact_O': observed_impact_O,
        'observed_impact_R': observed_impact_R,
        'observed_total_impact': observed_total_impact,
        'expected_utilities_O': expected_utilities_O,
        'expected_utilities_R': expected_utilities_R,
        'recalculated_utilities_O': recalculated_utilities_O,  # Add the recalculated utilities
        'recalculated_utilities_R': recalculated_utilities_R,  # Add the recalculated utilities
        'normalized_impact_O': normalized_impact_O,
        'normalized_impact_R': normalized_impact_R,
        'beliefs_O': beliefs_O,
        'beliefs_R': beliefs_R,
        'likelihood_O': likelihood_O,
        'likelihood_R': likelihood_R,
        'unnormalized_posterior_O': unnormalized_posterior_O,
        'unnormalized_posterior_R': unnormalized_posterior_R,
        'normalization_factor_O': normalization_factor_O,
        'normalization_factor_R': normalization_factor_R,
        'updated_beliefs_O': updated_beliefs_O,
        'updated_beliefs_R': updated_beliefs_R,
        'exigence_change': exigence_change
    }


# Calculate one step of the three-agent game theoretic model
def calculate_game_step_three_agents(params, beliefs_O_R=None, beliefs_O_M=None, 
                                    beliefs_R_O=None, beliefs_R_M=None,
                                    beliefs_M_O=None, beliefs_M_R=None, t=0):
    """
    Calculates one step of the game theoretic model with three agents:
    - O (Organization/University)
    - R (Researcher)
    - M (Museum)
    """
    strategies = get_strategies()
    
    # Get impact values for the scenario
    impact_O, impact_R, impact_M = get_impact_values_three_agents(params.scenario)
    
    # Normalize impacts according to equation 4
    normalized_impact_O = {k: v * params.M_O for k, v in impact_O.items()}
    normalized_impact_R = {k: v * params.M_R for k, v in impact_R.items()}
    normalized_impact_M = {k: v * params.M_M for k, v in impact_M.items()}
    
    # Initialize beliefs if not provided
    if beliefs_O_R is None or beliefs_O_M is None or beliefs_R_O is None or beliefs_R_M is None or beliefs_M_O is None or beliefs_M_R is None:
        sum_impact_O = sum(normalized_impact_O.values())
        sum_impact_R = sum(normalized_impact_R.values())
        sum_impact_M = sum(normalized_impact_M.values())
        
        # O's beliefs about R and M
        beliefs_O_R = {s: normalized_impact_R[s]/sum_impact_R for s in strategies}
        beliefs_O_M = {s: normalized_impact_M[s]/sum_impact_M for s in strategies}
        
        # R's beliefs about O and M
        beliefs_R_O = {s: normalized_impact_O[s]/sum_impact_O for s in strategies}
        beliefs_R_M = {s: normalized_impact_M[s]/sum_impact_M for s in strategies}
        
        # M's beliefs about O and R
        beliefs_M_O = {s: normalized_impact_O[s]/sum_impact_O for s in strategies}
        beliefs_M_R = {s: normalized_impact_R[s]/sum_impact_R for s in strategies}
    
    # Calculate expected impacts with three agents
    expected_impacts_O = {}
    expected_impacts_R = {}
    expected_impacts_M = {}
    
    for s_O in strategies:
        # Expected impact of O's strategy
        own_impact = normalized_impact_O[s_O]
        r_expected_impact = sum(beliefs_O_R[s_R] * normalized_impact_R[s_R] for s_R in strategies)
        m_expected_impact = sum(beliefs_O_M[s_M] * normalized_impact_M[s_M] for s_M in strategies)
        expected_impacts_O[s_O] = own_impact + r_expected_impact + m_expected_impact
    
    for s_R in strategies:
        # Expected impact of R's strategy
        own_impact = normalized_impact_R[s_R]
        o_expected_impact = sum(beliefs_R_O[s_O] * normalized_impact_O[s_O] for s_O in strategies)
        m_expected_impact = sum(beliefs_R_M[s_M] * normalized_impact_M[s_M] for s_M in strategies)
        expected_impacts_R[s_R] = own_impact + o_expected_impact + m_expected_impact
    
    for s_M in strategies:
        # Expected impact of M's strategy
        own_impact = normalized_impact_M[s_M]
        o_expected_impact = sum(beliefs_M_O[s_O] * normalized_impact_O[s_O] for s_O in strategies)
        r_expected_impact = sum(beliefs_M_R[s_R] * normalized_impact_R[s_R] for s_R in strategies)
        expected_impacts_M[s_M] = own_impact + o_expected_impact + r_expected_impact
    
    # Calculate expected utilities
    expected_utilities_O = {}
    expected_utilities_R = {}
    expected_utilities_M = {}
    
    for s_O in strategies:
        private_benefit = params.P_egoistic if s_O[0] == 'e' else params.P_altruistic
        expected_utilities_O[s_O] = params.alpha_O * expected_impacts_O[s_O] + (1 - params.alpha_O) * private_benefit
    
    for s_R in strategies:
        private_benefit = params.P_egoistic if s_R[0] == 'e' else params.P_altruistic
        expected_utilities_R[s_R] = params.alpha_R * expected_impacts_R[s_R] + (1 - params.alpha_R) * private_benefit
    
    for s_M in strategies:
        private_benefit = params.P_egoistic if s_M[0] == 'e' else params.P_altruistic
        expected_utilities_M[s_M] = params.alpha_M * expected_impacts_M[s_M] + (1 - params.alpha_M) * private_benefit
    
    # Choose optimal strategies
    s_O_opt = max(expected_utilities_O, key=expected_utilities_O.get)
    s_R_opt = max(expected_utilities_R, key=expected_utilities_R.get)
    s_M_opt = max(expected_utilities_M, key=expected_utilities_M.get)
    
    # Calculate observed impact with random variation
    import random
    
    # Apply small random variation to observed impacts
    observed_impact_O = min(max(normalized_impact_O[s_O_opt] + random.random() * 0.1 - 0.05, 0), params.M_O)
    observed_impact_R = min(max(normalized_impact_R[s_R_opt] + random.random() * 0.1 - 0.05, 0), params.M_R)
    observed_impact_M = min(max(normalized_impact_M[s_M_opt] + random.random() * 0.1 - 0.05, 0), params.M_M)
    
    # Calculate total observed impact
    observed_total_impact = observed_impact_O + observed_impact_R + observed_impact_M
    
    # Update beliefs (Bayesian update) for each agent
    # Calculate likelihoods
    likelihood_O_R = {}
    likelihood_O_M = {}
    likelihood_R_O = {}
    likelihood_R_M = {}
    likelihood_M_O = {}
    likelihood_M_R = {}
    
    for s in strategies:
        # O's beliefs update
        likelihood_O_R[s] = calculate_likelihood(s, observed_total_impact, normalized_impact_R, strategies)
        likelihood_O_M[s] = calculate_likelihood(s, observed_total_impact, normalized_impact_M, strategies)
        
        # R's beliefs update
        likelihood_R_O[s] = calculate_likelihood(s, observed_total_impact, normalized_impact_O, strategies)
        likelihood_R_M[s] = calculate_likelihood(s, observed_total_impact, normalized_impact_M, strategies)
        
        # M's beliefs update
        likelihood_M_O[s] = calculate_likelihood(s, observed_total_impact, normalized_impact_O, strategies)
        likelihood_M_R[s] = calculate_likelihood(s, observed_total_impact, normalized_impact_R, strategies)
    
    # Calculate unnormalized posterior beliefs
    unnormalized_posterior_O_R = {s: likelihood_O_R[s] * beliefs_O_R[s] for s in strategies}
    unnormalized_posterior_O_M = {s: likelihood_O_M[s] * beliefs_O_M[s] for s in strategies}
    unnormalized_posterior_R_O = {s: likelihood_R_O[s] * beliefs_R_O[s] for s in strategies}
    unnormalized_posterior_R_M = {s: likelihood_R_M[s] * beliefs_R_M[s] for s in strategies}
    unnormalized_posterior_M_O = {s: likelihood_M_O[s] * beliefs_M_O[s] for s in strategies}
    unnormalized_posterior_M_R = {s: likelihood_M_R[s] * beliefs_M_R[s] for s in strategies}
    
    # Calculate normalization factors
    normalization_factor_O_R = sum(unnormalized_posterior_O_R.values())
    normalization_factor_O_M = sum(unnormalized_posterior_O_M.values())
    normalization_factor_R_O = sum(unnormalized_posterior_R_O.values())
    normalization_factor_R_M = sum(unnormalized_posterior_R_M.values())
    normalization_factor_M_O = sum(unnormalized_posterior_M_O.values())
    normalization_factor_M_R = sum(unnormalized_posterior_M_R.values())
    
    # Calculate updated beliefs (normalized posterior)
    updated_beliefs_O_R = {s: v/normalization_factor_O_R for s, v in unnormalized_posterior_O_R.items()}
    updated_beliefs_O_M = {s: v/normalization_factor_O_M for s, v in unnormalized_posterior_O_M.items()}
    updated_beliefs_R_O = {s: v/normalization_factor_R_O for s, v in unnormalized_posterior_R_O.items()}
    updated_beliefs_R_M = {s: v/normalization_factor_R_M for s, v in unnormalized_posterior_R_M.items()}
    updated_beliefs_M_O = {s: v/normalization_factor_M_O for s, v in unnormalized_posterior_M_O.items()}
    updated_beliefs_M_R = {s: v/normalization_factor_M_R for s, v in unnormalized_posterior_M_R.items()}
    
    # Update exigence according to equation 10
    X_next = params.X_init + params.beta * (params.theta - observed_total_impact) * params.X_init * (1 - params.X_init)
    exigence_change = X_next - params.X_init
    
    return {
        'step': t,
        'X_t': params.X_init,
        'X_next': X_next,
        'strategy_O': s_O_opt,
        'strategy_R': s_R_opt,
        'strategy_M': s_M_opt,
        'observed_impact_O': observed_impact_O,
        'observed_impact_R': observed_impact_R,
        'observed_impact_M': observed_impact_M,
        'observed_total_impact': observed_total_impact,
        'expected_utilities_O': expected_utilities_O,
        'expected_utilities_R': expected_utilities_R,
        'expected_utilities_M': expected_utilities_M,
        'beliefs_O_R': beliefs_O_R,
        'beliefs_O_M': beliefs_O_M,
        'beliefs_R_O': beliefs_R_O,
        'beliefs_R_M': beliefs_R_M,
        'beliefs_M_O': beliefs_M_O,
        'beliefs_M_R': beliefs_M_R,
        'updated_beliefs_O_R': updated_beliefs_O_R,
        'updated_beliefs_O_M': updated_beliefs_O_M,
        'updated_beliefs_R_O': updated_beliefs_R_O,
        'updated_beliefs_R_M': updated_beliefs_R_M,
        'updated_beliefs_M_O': updated_beliefs_M_O,
        'updated_beliefs_M_R': updated_beliefs_M_R,
        'exigence_change': exigence_change
    }

# Simulate multiple steps with three agents
def simulate_multi_step_three_agents(params, steps=10):
    """Simulates multiple steps of the three-agent model"""
    results = []
    
    # Initial step
    current_beliefs_O_R = None
    current_beliefs_O_M = None
    current_beliefs_R_O = None
    current_beliefs_R_M = None
    current_beliefs_M_O = None
    current_beliefs_M_R = None
    current_X = params.X_init
    
    for t in range(steps):
        # Set current exigence
        params.X_init = current_X
        
        # Calculate current step
        step_result = calculate_game_step_three_agents(
            params, 
            beliefs_O_R=current_beliefs_O_R, 
            beliefs_O_M=current_beliefs_O_M,
            beliefs_R_O=current_beliefs_R_O, 
            beliefs_R_M=current_beliefs_R_M,
            beliefs_M_O=current_beliefs_M_O, 
            beliefs_M_R=current_beliefs_M_R,
            t=t
        )
        
        # Record results
        results.append({
            'step': t,
            'X_t': current_X,
            'X_next': step_result['X_next'],
            'strategy_O': step_result['strategy_O'],
            'strategy_R': step_result['strategy_R'],
            'strategy_M': step_result['strategy_M'],
            'observed_impact_O': step_result['observed_impact_O'],
            'observed_impact_R': step_result['observed_impact_R'],
            'observed_impact_M': step_result['observed_impact_M'],
            'observed_total_impact': step_result['observed_total_impact'],
            'utility_O': step_result['expected_utilities_O'][step_result['strategy_O']],
            'utility_R': step_result['expected_utilities_R'][step_result['strategy_R']],
            'utility_M': step_result['expected_utilities_M'][step_result['strategy_M']]
        })
        
        # Update for next step
        current_X = step_result['X_next']
        current_beliefs_O_R = step_result['updated_beliefs_O_R']
        current_beliefs_O_M = step_result['updated_beliefs_O_M']
        current_beliefs_R_O = step_result['updated_beliefs_R_O']
        current_beliefs_R_M = step_result['updated_beliefs_R_M']
        current_beliefs_M_O = step_result['updated_beliefs_M_O']
        current_beliefs_M_R = step_result['updated_beliefs_M_R']
    
    return pd.DataFrame(results)

# Simulate multiple steps of the model
def simulate_multi_step(params, steps=10, export_excel=False):
    """
    Simulates multiple steps of the model, updating beliefs and exigence at each step
    """
    results = []
    
    # Initial step
    current_beliefs_O = None
    current_beliefs_R = None
    current_X = params.X_init
    
    for t in range(steps):
        # Set current exigence
        params.X_init = current_X
        
        # Store previous beliefs for Excel output (if not initial step)
        prev_beliefs_O = current_beliefs_O if t > 0 else None
        prev_beliefs_R = current_beliefs_R if t > 0 else None
        
        # Calculate current step
        step_result = calculate_game_step(params, beliefs_O=current_beliefs_O, beliefs_R=current_beliefs_R, 
                                         t=t, export_excel=export_excel,
                                         prev_beliefs_O=prev_beliefs_O, prev_beliefs_R=prev_beliefs_R)
        
        # Record results
        results.append({
            'step': t,
            'X_t': current_X,
            'X_next': step_result['X_next'],
            'strategy_O': step_result['strategy_O'],
            'strategy_R': step_result['strategy_R'],
            'observed_impact': step_result['observed_total_impact'],
            'utility_O': step_result['expected_utilities_O'][step_result['strategy_O']],
            'utility_R': step_result['expected_utilities_R'][step_result['strategy_R']]
        })
        
        # Update for next step
        current_X = step_result['X_next']
        current_beliefs_O = step_result['updated_beliefs_O']
        current_beliefs_R = step_result['updated_beliefs_R']
    
    return pd.DataFrame(results)
    
# Function to create visualization for three-agent scenario
def visualize_three_agent_scenario(save_pdf=True):
    """
    Create visualizations for the three-agent Citizen Science scenario with Museum
    """
    # Set up parameters
    params = ModelParams('citizen_science_museum')
    
    # Get impact values
    impact_O, impact_R, impact_M = get_impact_values_three_agents('citizen_science_museum')
    
    # Simulate 15 steps
    df = simulate_multi_step_three_agents(params, steps=15)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Plot impact matrices for all three actors
    plt.subplot(2, 3, 1)
    strategies = get_strategies()
    impact_values_O = np.array([impact_O[s] for s in strategies]).reshape(2, 3)
    plt.imshow(impact_values_O, cmap='Blues', vmin=0, vmax=1)
    plt.title('Organization Impact')
    plt.xticks(np.arange(3), ['k', 'd', 'p'])
    plt.yticks(np.arange(2), ['a', 'e'])
    plt.xlabel('Mode')
    plt.ylabel('Role')
    plt.colorbar(label='Impact')
    
    # Add text annotations
    for i in range(2):
        for j in range(3):
            s = (('a', 'e')[i], ('k', 'd', 'p')[j])
            plt.text(j, i, f"{impact_O[s]:.1f}", ha="center", va="center", 
                    color="white" if impact_O[s] > 0.5 else "black")
    
    plt.subplot(2, 3, 2)
    impact_values_R = np.array([impact_R[s] for s in strategies]).reshape(2, 3)
    plt.imshow(impact_values_R, cmap='Blues', vmin=0, vmax=1)
    plt.title('Researcher Impact')
    plt.xticks(np.arange(3), ['k', 'd', 'p'])
    plt.yticks(np.arange(2), ['a', 'e'])
    plt.xlabel('Mode')
    plt.ylabel('Role')
    plt.colorbar(label='Impact')
    
    # Add text annotations
    for i in range(2):
        for j in range(3):
            s = (('a', 'e')[i], ('k', 'd', 'p')[j])
            plt.text(j, i, f"{impact_R[s]:.1f}", ha="center", va="center", 
                    color="white" if impact_R[s] > 0.5 else "black")
    
    plt.subplot(2, 3, 3)
    impact_values_M = np.array([impact_M[s] for s in strategies]).reshape(2, 3)
    plt.imshow(impact_values_M, cmap='Blues', vmin=0, vmax=1)
    plt.title('Museum Impact')
    plt.xticks(np.arange(3), ['k', 'd', 'p'])
    plt.yticks(np.arange(2), ['a', 'e'])
    plt.xlabel('Mode')
    plt.ylabel('Role')
    plt.colorbar(label='Impact')
    
    # Add text annotations
    for i in range(2):
        for j in range(3):
            s = (('a', 'e')[i], ('k', 'd', 'p')[j])
            plt.text(j, i, f"{impact_M[s]:.1f}", ha="center", va="center", 
                    color="white" if impact_M[s] > 0.5 else "black")
    
    # 2. Plot exigence and impact over time
    plt.subplot(2, 2, 3)
    plt.plot(df['step'], df['X_t'], 'k-o', label='Exigence')
    plt.plot(df['step'], df['observed_impact_O'], 'b-o', label='Org Impact')
    plt.plot(df['step'], df['observed_impact_R'], 'r-o', label='Res Impact')
    plt.plot(df['step'], df['observed_impact_M'], 'g-o', label='Museum Impact')
    plt.plot(df['step'], df['observed_total_impact'], 'm-o', label='Total Impact')
    plt.axhline(y=params.theta, color='k', linestyle='--', label=f'Threshold (θ={params.theta})')
    plt.title('Exigence and Impact Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # 3. Plot strategy choices over time
    plt.subplot(2, 2, 4)
    # Create a mapping of strategies to integers for plotting
    strategy_dict = {s: i for i, s in enumerate(strategies)}
    
    plt.plot(df['step'], [strategy_dict[s] for s in df['strategy_O']], 'bo-', label='Org Strategy')
    plt.plot(df['step'], [strategy_dict[s] for s in df['strategy_R']], 'ro-', label='Res Strategy')
    plt.plot(df['step'], [strategy_dict[s] for s in df['strategy_M']], 'go-', label='Museum Strategy')
    
    # Set y-ticks to strategy labels
    plt.yticks(range(len(strategies)), [f'{s[0]},{s[1]}' for s in strategies])
    plt.title('Strategy Choices Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Strategy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    if save_pdf:
        plt.savefig('simulations_3agent/three_agent_citizen_science.pdf', format='pdf')
        print("Saved visualization as 'simulations_3agent/three_agent_citizen_science.pdf'")
    
    return fig, df


# Function to compare two-agent and three-agent scenarios
# Function to compare two-agent and three-agent scenarios
def compare_scenarios(save_pdf=True):
    """
    Compare the original two-agent scenario with the three-agent scenario
    """
    
    # Set up parameters
    orig_params = OrigModelParams('citizen_science')
    new_params = ModelParams('citizen_science_museum')
    
    # Simulate both models
    orig_df = simulate_multi_step(orig_params, steps=15)
    new_df = simulate_multi_step_three_agents(new_params, steps=15)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Compare exigence evolution
    plt.subplot(2, 2, 1)
    plt.plot(orig_df['step'], orig_df['X_t'], 'b-o', label='Two-Agent Model')
    plt.plot(new_df['step'], new_df['X_t'], 'r-o', label='Three-Agent Model')
    plt.title('Exigence Evolution Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Exigence')
    plt.legend()
    plt.grid(True)
    
    # 2. Compare total impact
    plt.subplot(2, 2, 2)
    # The column name is 'observed_impact' in the two-agent model, not 'observed_total_impact'
    plt.plot(orig_df['step'], orig_df['observed_impact'], 'b-o', label='Two-Agent Total Impact')
    plt.plot(new_df['step'], new_df['observed_total_impact'], 'r-o', label='Three-Agent Total Impact')
    plt.axhline(y=orig_params.theta, color='k', linestyle='--', label=f'Threshold (θ={orig_params.theta})')
    plt.title('Total Impact Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Impact')
    plt.legend()
    plt.grid(True)
    
    # 3. Compare organization strategies
    plt.subplot(2, 2, 3)
    strategies = get_strategies()
    strategy_dict = {s: i for i, s in enumerate(strategies)}
    
    plt.plot(orig_df['step'], [strategy_dict[s] for s in orig_df['strategy_O']], 'b-o', label='Org (Two-Agent)')
    plt.plot(orig_df['step'], [strategy_dict[s] for s in orig_df['strategy_R']], 'b--o', label='Res (Two-Agent)')
    plt.plot(new_df['step'], [strategy_dict[s] for s in new_df['strategy_O']], 'r-o', label='Org (Three-Agent)')
    plt.plot(new_df['step'], [strategy_dict[s] for s in new_df['strategy_R']], 'r--o', label='Res (Three-Agent)')
    
    # Set y-ticks to strategy labels
    plt.yticks(range(len(strategies)), [f'{s[0]},{s[1]}' for s in strategies])
    plt.title('Strategy Comparison (Original Actors)')
    plt.xlabel('Time Steps')
    plt.ylabel('Strategy')
    plt.legend()
    plt.grid(True)
    
    # 4. Distribution of impacts in three-agent model
    plt.subplot(2, 2, 4)
    
    # Create data for stacked bar chart
    steps = range(len(new_df))
    impact_O = new_df['observed_impact_O']
    impact_R = new_df['observed_impact_R']
    impact_M = new_df['observed_impact_M']
    
    # Create stacked bar chart
    plt.bar(steps, impact_O, label='Organization')
    plt.bar(steps, impact_R, bottom=impact_O, label='Researcher')
    plt.bar(steps, impact_M, bottom=[impact_O[i] + impact_R[i] for i in range(len(impact_O))], label='Museum')
    
    plt.title('Impact Distribution in Three-Agent Model')
    plt.xlabel('Time Steps')
    plt.ylabel('Impact')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    if save_pdf:
        plt.savefig('simulations_3agent/scenario_comparison.pdf', format='pdf')
        print("Saved comparison as 'simulations_3agent/scenario_comparison.pdf'")
    
    return fig, orig_df, new_df

# Run simulation and analysis
# Run simulation and analysis
if __name__ == "__main__":
    print("Running three-agent Citizen Science simulation with Museum...")

    # Set the current working directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(SCRIPT_DIR)
    print("Current working directory:", os.getcwd())
        
    # Create simulations directory if it doesn't exist
    if not os.path.exists('simulations_3agent'):
        os.makedirs('simulations_3agent')

    # Create visualization
    fig, simulation_results = visualize_three_agent_scenario(save_pdf=True)
    
    # Print summary of results
    print("\nSummary of Three-Agent Citizen Science Simulation:")
    print(f"Final strategies: Organization={simulation_results['strategy_O'].iloc[-1]}, " +
          f"Researcher={simulation_results['strategy_R'].iloc[-1]}, " + 
          f"Museum={simulation_results['strategy_M'].iloc[-1]}")
    print(f"Final exigence value: {simulation_results['X_t'].iloc[-1]:.4f}")
    print(f"Final impact values: Organization={simulation_results['observed_impact_O'].iloc[-1]:.4f}, " +
          f"Researcher={simulation_results['observed_impact_R'].iloc[-1]:.4f}, " +
          f"Museum={simulation_results['observed_impact_M'].iloc[-1]:.4f}")
    print(f"Final total impact: {simulation_results['observed_total_impact'].iloc[-1]:.4f}")
    
    # Print the most common strategies
    print("\nMost common strategies:")
    print(f"Organization: {simulation_results['strategy_O'].mode().iloc[0]}")
    print(f"Researcher: {simulation_results['strategy_R'].mode().iloc[0]}")
    print(f"Museum: {simulation_results['strategy_M'].mode().iloc[0]}")
    
    # Calculate average impacts
    print("\nAverage impacts:")
    print(f"Organization: {simulation_results['observed_impact_O'].mean():.4f}")
    print(f"Researcher: {simulation_results['observed_impact_R'].mean():.4f}")
    print(f"Museum: {simulation_results['observed_impact_M'].mean():.4f}")
    print(f"Total: {simulation_results['observed_total_impact'].mean():.4f}")
    
    # Also run comparison with two-agent model
    print("\nComparing two-agent and three-agent models...")
    compare_fig, orig_df, new_df = compare_scenarios(save_pdf=True)
    
    # Print comparison summary
    print("\nTwo-Agent Model Final State:")
    print(f"Final strategies: Organization={orig_df['strategy_O'].iloc[-1]}, Researcher={orig_df['strategy_R'].iloc[-1]}")
    print(f"Final exigence value: {orig_df['X_t'].iloc[-1]:.4f}")
    print(f"Final total impact: {orig_df['observed_impact'].iloc[-1]:.4f}")
    
    print("\nThree-Agent Model Final State (for comparison):")
    print(f"Final strategies: Organization={new_df['strategy_O'].iloc[-1]}, Researcher={new_df['strategy_R'].iloc[-1]}, Museum={new_df['strategy_M'].iloc[-1]}")
    print(f"Final exigence value: {new_df['X_t'].iloc[-1]:.4f}")
    print(f"Final total impact: {new_df['observed_total_impact'].iloc[-1]:.4f}")
    
    # Save figures
    plt.close(fig)  # Close the first figure to ensure it's saved
    plt.close(compare_fig)  # Close the comparison figure to ensure it's saved