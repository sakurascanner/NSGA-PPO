import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties
from botorch.acquisition.monte_carlo import qExpectedImprovement
from AMP_NMCF import BoPpoAMPNMCFEnv

def evaluate_circuit_for_bo(parameters):
    """
    Wrapper function to evaluate a parameter dict using your SPICE setup.
    """
    # Convert dict to numpy array matching your action space order
    action_array = np.array([parameters[f"param_{i}"] for i in range(24)])
    
    # Initialize a temporary env just for evaluation
    env = BoPpoAMPNMCFEnv() 
    op_results, sim_results, Ib = env.do_simulation(action_array)
    
    # The paper's BO goal: maximize performance toward the mean of the spec space.
    # We can use the existing composite reward, or a specific heuristic.
    # Here we use your existing reward function.
    try:
        info = env._get_info(sim_results)
        reward = float(info['reward'])
    except:
        reward = -11.0
    return {"composite_reward": (reward, 0.0)}  # Ê¼ÖÕ·µ»Ø tuple

def run_bo_vanguard(env_bounds_low, env_bounds_high, max_simulations=50, int_indices=None):
    """
    Runs Bayesian Optimization to find the optimal starting point x_0.
    """
    if int_indices is None:
        int_indices = []

    ax_client = AxClient()
    
    # Define search space dynamically
    parameters = [
        {
            "name": f"param_{i}",
            "type": "range",
            # Cast bounds to int if it's an integer parameter, otherwise float
            "bounds": [
                int(env_bounds_low[i]) if i in int_indices else float(env_bounds_low[i]),
                int(env_bounds_high[i]) if i in int_indices else float(env_bounds_high[i])
            ],
            "value_type": "int" if i in int_indices else "float"
        }
        for i in range(len(env_bounds_low))
    ]
    
    ax_client.create_experiment(
        name="RoSE_Opt_BO",
        parameters=parameters,
        objectives={"composite_reward": ObjectiveProperties(minimize=False)},
    )
    
    # Optimization loop
    for i in range(max_simulations):
        parameters, trial_index = ax_client.get_next_trial()
        results = evaluate_circuit_for_bo(parameters)
        ax_client.complete_trial(trial_index=trial_index, raw_data=results)
        
    best_parameters, metrics = ax_client.get_best_parameters()
    best_action_array = np.array([best_parameters[f"param_{i}"] for i in range(len(env_bounds_low))])
    
    return best_action_array