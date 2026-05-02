import numpy as np
from scipy.integrate import quad
import random
import matplotlib.pyplot as plt
import pandas as pd
import sys

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PARAMETERS & PAPER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
NUM_MACHINES   = 6
NUM_BUFFERS    = 5
BOTTLENECK     = 3   # M4 (0-indexed)

CYCLE_TIME        = [2.1, 2.8, 2.3, 3.5, 2.3, 2.5]
INIT_VIRTUAL_AGE  = [360, 480, 550,  80, 200, 100]
POWER_PROCESSING  = [450, 360, 240, 288, 360, 360]
POWER_STANDBY     = [300, 240, 160, 185, 240, 240]
POWER_MAINTENANCE = [675, 540, 360, 432, 540, 540]

# Paper typo correction: M1 eta 82.2 -> 822, M2 108.5 -> 1085
# This ensures a0 < eta for all machines, preventing immediate infinite failures
WEIBULL_ETA       = [822, 1085, 2001, 1412, 397, 256]
WEIBULL_BETA      = [4.95, 4.51,  1.52, 1.75, 2.91, 3.82]

BUFFER_CAPACITY = [8, 4, 5, 5, 4]
BUFFER_INIT     = [2, 2, 2, 2, 2]

SIM_DURATION        = 2400
DECISION_STEP       = 1
MIN_SWITCH_INTERVAL = 10

PRODUCT_VALUE     = 300.0
ELECTRICITY_PRICE = 0.64 / 60.0  # yuan per kW-minute

MAINT_DURATION   = [0,  5,  20,  40]
MAINT_FIXED_COST = [0, 100, 800, 2000]
MAINT_AGE_FACTOR = [0,  1.0, 0.6, 0.0]

GA_POPULATION_SIZE = 6
GA_CROSSOVER_PROB  = 0.8
GA_MUTATION_PROB   = 0.005

STATE_IDLE       = 'IDLE'
STATE_PROCESSING = 'PROCESSING'
STATE_BLOCKED    = 'BLOCKED'
STATE_STARVED    = 'STARVED'
STATE_DOWN       = 'DOWN'


# ─────────────────────────────────────────────────────────────────────────────
# WEIBULL & MAINTENANCE LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def sample_next_failure(eta, beta, virtual_age=0.0):
    """
    Conditional Weibull inverse-CDF: t* = eta*((a/eta)^b - ln U)^(1/b) - a
    """
    U = random.uniform(1e-9, 1.0 - 1e-9)
    a = min(virtual_age, 0.95 * eta)  # clamp to valid Weibull range
    cum_hazard = (a / eta) ** beta
    return max(0.0, eta * (cum_hazard - np.log(U)) ** (1.0 / beta) - a)

def expected_maintenance_cost_rate(maint_level, virtual_age_before, runnable_time, machine_idx, opportunity_window):
    m    = maint_level
    eta  = WEIBULL_ETA[machine_idx]
    beta = WEIBULL_BETA[machine_idx]
    dm   = MAINT_DURATION[m]
    cm   = MAINT_FIXED_COST[m]
    Am   = MAINT_AGE_FACTOR[m]

    a_after   = Am * (virtual_age_before + runnable_time)
    a_after   = min(a_after, 0.95 * eta)  # Match simulation clamp
    prod_loss = max((dm - opportunity_window) / CYCLE_TIME[machine_idx], 0.0)
    C_maint   = cm + PRODUCT_VALUE * prod_loss

    R_a = np.exp(-((a_after / eta) ** beta)) if eta > 0 else 0.0
    if R_a < 1e-300:
        mean_residual = 0.0
    else:
        def surv_integrand(t):
            return np.exp(-((a_after + t) / eta) ** beta) / R_a
        mean_residual, _ = quad(surv_integrand, 0, 10 * eta, limit=100)

    return C_maint / max(dm + mean_residual, 1e-9)

def decide_maintenance_level(machine_idx, virtual_age_before, runnable_time, opportunity_window):
    best_level, best_rate = 1, float('inf')
    for m in [1, 2, 3]:
        rate = expected_maintenance_cost_rate(m, virtual_age_before, runnable_time, machine_idx, opportunity_window)
        if rate < best_rate:
            best_rate, best_level = rate, m
    return best_level


# ─────────────────────────────────────────────────────────────────────────────
# ENERGY & CACHE LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def calculate_cache_potential(line, machine_idx):
    if machine_idx < BOTTLENECK:
        return sum(line.buffer_levels[k] for k in range(machine_idx, BOTTLENECK))
    elif machine_idx > BOTTLENECK:
        return sum(BUFFER_CAPACITY[k] - line.buffer_levels[k] for k in range(BOTTLENECK, machine_idx))
    return 0.0

def calculate_opportunity_window(line, machine_idx):
    if machine_idx == BOTTLENECK:
        return 0.0
    epsilon = calculate_cache_potential(line, machine_idx)
    return epsilon * CYCLE_TIME[BOTTLENECK]

def cost_processing_energy(states, controls, maint):
    cost = 0.0
    for i in range(NUM_MACHINES):
        if maint[i] == 1 and controls[i] == 1:
            if states[i] == STATE_PROCESSING:
                cost += POWER_PROCESSING[i] * ELECTRICITY_PRICE * DECISION_STEP
            else:
                cost += POWER_STANDBY[i] * ELECTRICITY_PRICE * DECISION_STEP
    return cost

def cost_state_transition(curr_ctrl, prev_ctrl, phi):
    cost = 0.0
    for i in range(NUM_MACHINES):
        if curr_ctrl[i] != prev_ctrl[i]:
            cost += max(0.0, 15.0 - phi[i]) * ELECTRICITY_PRICE
    return cost


# ─────────────────────────────────────────────────────────────────────────────
# GENETIC ALGORITHM FOR ENERGY CONTROL
# ─────────────────────────────────────────────────────────────────────────────
def genetic_algorithm_energy(line):
    pop = []
    pop.append(list(line.control_inputs))
    while len(pop) < GA_POPULATION_SIZE:
        ind = []
        for i in range(NUM_MACHINES):
            if line.maint_inputs[i] == 0:
                ind.append(0)
            elif line.phi_hold[i] < MIN_SWITCH_INTERVAL:
                ind.append(line.control_inputs[i])
            else:
                ind.append(random.choice([0, 1]))
        pop.append(ind)

    def fitness(ind):
        ce = cost_processing_energy(line.machine_states, ind, line.maint_inputs)
        csw = cost_state_transition(ind, line.control_inputs, line.phi_hold)
        
        c_prod = 0.0
        upstream_off = 0
        downstream_off = 0
        for i in range(NUM_MACHINES):
            if ind[i] == 0 and line.maint_inputs[i] == 1:
                if i < BOTTLENECK: upstream_off += 1
                elif i > BOTTLENECK: downstream_off += 1
                
                ow = calculate_opportunity_window(line, i)
                if line.control_inputs[i] == 1:
                    loss = max(0.0, MIN_SWITCH_INTERVAL * 2.5 - ow)
                else:
                    loss = max(0.0, line.phi_hold[i] + DECISION_STEP - ow)
                c_prod += PRODUCT_VALUE * (loss / CYCLE_TIME[i])
                
        if upstream_off > 1 or downstream_off > 1:
            c_prod += 1000000.0  # Eq 40-41 constraint
            
        return -(ce + csw + c_prod)

    for generation in range(5):
        fits = [fitness(ind) for ind in pop]
        min_f = min(fits)
        norm_fits = [f - min_f + 0.01 for f in fits]
        total_f = sum(norm_fits)
        probs = [f / total_f for f in norm_fits]

        new_pop = [pop[np.argmax(fits)]]
        while len(new_pop) < GA_POPULATION_SIZE:
            p1, p2 = random.choices(pop, weights=probs, k=2)
            c1, c2 = list(p1), list(p2)
            if random.random() < GA_CROSSOVER_PROB:
                pt = random.randint(1, NUM_MACHINES - 1)
                c1 = p1[:pt] + p2[pt:]
                c2 = p2[:pt] + p1[pt:]
            for c in (c1, c2):
                if random.random() < GA_MUTATION_PROB:
                    mut_idx = random.randint(0, NUM_MACHINES - 1)
                    if line.maint_inputs[mut_idx] == 1 and line.phi_hold[mut_idx] >= MIN_SWITCH_INTERVAL:
                        c[mut_idx] = 1 - c[mut_idx]
                if len(new_pop) < GA_POPULATION_SIZE:
                    new_pop.append(c)
        pop = new_pop

    best_ind = pop[np.argmax([fitness(ind) for ind in pop])]
    return best_ind


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class ProductionLine:
    def __init__(self):
        self.machine_states     = [STATE_IDLE] * NUM_MACHINES
        self.buffer_levels      = list(BUFFER_INIT)
        self.control_inputs     = [1] * NUM_MACHINES
        self.maint_inputs       = [1] * NUM_MACHINES
        self.cumulative_output  = [0] * NUM_MACHINES
        self.virtual_ages       = list(INIT_VIRTUAL_AGE)
        self.time_to_failure    = [sample_next_failure(WEIBULL_ETA[i], WEIBULL_BETA[i], INIT_VIRTUAL_AGE[i]) for i in range(NUM_MACHINES)]
        self.remaining_proc     = list(CYCLE_TIME)
        
        self.age_at_last_maint  = list(INIT_VIRTUAL_AGE)
        self.time_since_maint   = [0.0] * NUM_MACHINES
        self.maint_time_remain  = [0.0] * NUM_MACHINES
        self.phi_hold           = [0.0] * NUM_MACHINES

        self.total_energy_cost  = 0.0
        self.total_switch_cost  = 0.0
        self.total_maint_cost   = 0.0
        self.processing_hours   = [0.0] * NUM_MACHINES
        self.standby_hours      = [0.0] * NUM_MACHINES

def run_simulation(scenario, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    opt_e = scenario in ('energy_only', 'dt_reo')
    opt_m = scenario in ('maint_only',  'dt_reo')

    line = ProductionLine()
    for t in range(SIM_DURATION):
        prev_ctrl  = list(line.control_inputs)
        phi_before = list(line.phi_hold)

        # 1. Failure & Maintenance
        for i in range(NUM_MACHINES):
            if line.machine_states[i] != STATE_DOWN:
                if line.control_inputs[i] == 1 and line.maint_inputs[i] == 1:
                    line.time_to_failure[i] -= DECISION_STEP
                    if line.time_to_failure[i] <= 0:
                        line.machine_states[i] = STATE_DOWN
                        line.maint_inputs[i] = 0
                        line.control_inputs[i] = 0
                        
                        OW = calculate_opportunity_window(line, i)
                        if opt_m:
                            m_level = decide_maintenance_level(i, line.virtual_ages[i], line.time_since_maint[i], OW)
                        else:
                            m_level = 1  # No decision -> Level 1

                        line.maint_time_remain[i] = MAINT_DURATION[m_level]
                        line.virtual_ages[i] = MAINT_AGE_FACTOR[m_level] * (line.virtual_ages[i] + line.time_since_maint[i])
                        line.total_maint_cost += MAINT_FIXED_COST[m_level] + PRODUCT_VALUE * max((MAINT_DURATION[m_level] - OW)/CYCLE_TIME[i], 0.0)
            else:
                line.maint_time_remain[i] -= DECISION_STEP
                if line.maint_time_remain[i] <= 0:
                    line.age_at_last_maint[i] = line.virtual_ages[i]
                    line.time_since_maint[i]  = 0.0
                    line.time_to_failure[i]   = sample_next_failure(WEIBULL_ETA[i], WEIBULL_BETA[i], line.virtual_ages[i])
                    line.maint_inputs[i]   = 1
                    line.machine_states[i] = STATE_IDLE
                    line.control_inputs[i] = 1

        # 2. Energy Optimization
        if opt_e:
            line.control_inputs = genetic_algorithm_energy(line)

        # 3. State update & Processing
        for i in range(NUM_MACHINES):
            if line.control_inputs[i] == prev_ctrl[i]:
                line.phi_hold[i] += DECISION_STEP
            else:
                line.phi_hold[i] = DECISION_STEP
                
            if line.maint_inputs[i] == 0 or line.control_inputs[i] == 0:
                continue
                
            starved = (i > 0 and line.buffer_levels[i-1] == 0)
            blocked = (i < NUM_MACHINES - 1 and line.buffer_levels[i] >= BUFFER_CAPACITY[i])
            
            if starved:
                line.machine_states[i] = STATE_STARVED
                line.standby_hours[i] += DECISION_STEP
            elif blocked:
                line.machine_states[i] = STATE_BLOCKED
                line.standby_hours[i] += DECISION_STEP
            else:
                line.machine_states[i] = STATE_PROCESSING
                line.processing_hours[i] += DECISION_STEP
                line.remaining_proc[i] -= DECISION_STEP
                line.virtual_ages[i] += DECISION_STEP
                line.time_since_maint[i] += DECISION_STEP
                
                if line.remaining_proc[i] <= 0:
                    line.cumulative_output[i] += 1
                    line.remaining_proc[i] = CYCLE_TIME[i]
                    if i > 0: line.buffer_levels[i-1] -= 1
                    if i < NUM_MACHINES - 1: line.buffer_levels[i] += 1

        # 4. Energy Costs
        line.total_energy_cost += cost_processing_energy(line.machine_states, line.control_inputs, line.maint_inputs)
        line.total_switch_cost += cost_state_transition(line.control_inputs, prev_ctrl, phi_before)
        for i in range(NUM_MACHINES):
            if line.maint_inputs[i] == 0:
                line.total_energy_cost += POWER_MAINTENANCE[i] * ELECTRICITY_PRICE * DECISION_STEP

    return line

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS & PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def run_average(scenario, n_runs):
    outputs, m_costs, e_costs, utils = [], [], [], []
    for i in range(n_runs):
        line = run_simulation(scenario, seed=42+i)
        
        # Corrected output per paper
        out = line.cumulative_output[-1] + sum(line.buffer_levels[k] * ((k+1)/NUM_MACHINES) for k in range(NUM_BUFFERS))
        outputs.append(out)
        m_costs.append(line.total_maint_cost)
        e_costs.append(line.total_energy_cost + line.total_switch_cost)
        
        u = [100.0 * line.processing_hours[k] / max(1e-9, line.processing_hours[k] + line.standby_hours[k]) for k in range(NUM_MACHINES)]
        utils.append(u)
        
    return {
        'output': np.mean(outputs),
        'maint_cost': np.mean(m_costs),
        'energy_cost': np.mean(e_costs),
        'utils': np.mean(utils, axis=0)
    }

def print_summary(results):
    print("\\n" + "="*80)
    print(f"{'Scenario':<20} {'Avg Util%':>10} {'Output':>10} {'Maint Cost':>12} {'Energy Cost':>12} {'Total Profit':>14}")
    print("="*80)
    for sc, res in results.items():
        avg_util = np.mean(res['utils'])
        profit = res['output'] * PRODUCT_VALUE - res['maint_cost'] - res['energy_cost']
        print(f"{sc:<20} {avg_util:>10.2f} {res['output']:>10.2f} {res['maint_cost']:>12.2f} {res['energy_cost']:>12.2f} {profit:>14.2f}")
    print("="*80)
    
    # Print comparison
    if 'dt_reo' in results and 'no_decision' in results:
        dr = results['dt_reo']
        nd = results['no_decision']
        
        dr_profit = dr['output'] * PRODUCT_VALUE - dr['maint_cost'] - dr['energy_cost']
        nd_profit = nd['output'] * PRODUCT_VALUE - nd['maint_cost'] - nd['energy_cost']
        
        dr_util = np.mean(dr['utils'])
        nd_util = np.mean(nd['utils'])
        
        print("\nDT-REO vs No Decision:")
        print(f"  Avg Machine Util : {dr_util:.2f}% vs {nd_util:.2f}%  [{dr_util - nd_util:+.2f} pp]")
        print(f"  Corrected Output : {dr['output']:.2f} vs {nd['output']:.2f}  [{(dr['output'] - nd['output']) / max(1e-9, nd['output']) * 100:+.2f}%]")
        print(f"  Maint Cost       : {dr['maint_cost']:.2f} vs {nd['maint_cost']:.2f}  [{(dr['maint_cost'] - nd['maint_cost']) / max(1e-9, nd['maint_cost']) * 100:+.2f}%]")
        print(f"  Energy Cost      : {dr['energy_cost']:.2f} vs {nd['energy_cost']:.2f}  [{(dr['energy_cost'] - nd['energy_cost']) / max(1e-9, nd['energy_cost']) * 100:+.2f}%]")
        print(f"  Total Profit     : {dr_profit:.2f} vs {nd_profit:.2f}  [{(dr_profit - nd_profit) / max(1e-9, nd_profit) * 100:+.2f}%]")
        print("="*80)

def plot_results(results):
    scenarios = list(results.keys())
    
    # 1. Machine Characteristics / Utilization Plot
    plt.figure(figsize=(10, 6))
    x = np.arange(NUM_MACHINES)
    width = 0.2
    
    for i, sc in enumerate(scenarios):
        plt.bar(x + i*width, results[sc]['utils'], width, label=sc)
    
    plt.title('Machine Utilization Comparison (Fig. 16)', fontsize=14, fontweight='bold')
    plt.xlabel('Machine', fontsize=12)
    plt.ylabel('Utilization (%)', fontsize=12)
    plt.xticks(x + width*1.5, [f'M{i+1}' for i in range(NUM_MACHINES)])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('utilization_comparison.png', dpi=300)
    print("Saved 'utilization_comparison.png'")

    # 2. Cost Distributions Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    profits = [results[sc]['output'] * PRODUCT_VALUE - results[sc]['maint_cost'] - results[sc]['energy_cost'] for sc in scenarios]
    m_costs = [results[sc]['maint_cost'] for sc in scenarios]
    e_costs = [results[sc]['energy_cost'] for sc in scenarios]
    
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    
    ax1.bar(scenarios, m_costs, color=colors)
    ax1.set_title('Maintenance Cost', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(scenarios, e_costs, color=colors)
    ax2.set_title('Energy Cost', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    ax3.bar(scenarios, profits, color=colors)
    ax3.set_title('Total Profit', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('cost_distributions.png', dpi=300)
    print("Saved 'cost_distributions.png'")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global NUM_MACHINES, BOTTLENECK, SIM_DURATION, DECISION_STEP
    
    print("\\n" + "=" * 62)
    print("  DT-REO: Digital Twin Real-Time Energy Optimization Model")
    print("=" * 62)
    print("  Configure your simulation parameters:\\n")

    def ask_int(prompt, default):
        raw = input(f"  {prompt} [default {default}]: ").strip()
        return int(raw) if raw else default

    NUM_MACHINES   = ask_int("Number of machines (2-6)", 6)
    NUM_MACHINES   = max(2, min(NUM_MACHINES, 6))
    
    BOTTLENECK_IN  = ask_int(f"Bottleneck machine number 1-{NUM_MACHINES}", 4)
    BOTTLENECK     = max(1, min(BOTTLENECK_IN, NUM_MACHINES)) - 1
    
    # Ensure the user's chosen bottleneck is genuinely the physical bottleneck
    global CYCLE_TIME, INIT_VIRTUAL_AGE, POWER_PROCESSING, POWER_STANDBY, POWER_MAINTENANCE, WEIBULL_ETA, WEIBULL_BETA
    physical_bn = max(range(NUM_MACHINES), key=lambda i: CYCLE_TIME[i])
    if BOTTLENECK != physical_bn:
        def swap(lst):
            lst[BOTTLENECK], lst[physical_bn] = lst[physical_bn], lst[BOTTLENECK]
        swap(CYCLE_TIME)
        swap(INIT_VIRTUAL_AGE)
        swap(POWER_PROCESSING)
        swap(POWER_STANDBY)
        swap(POWER_MAINTENANCE)
        swap(WEIBULL_ETA)
        swap(WEIBULL_BETA)
        
    SIM_DURATION   = ask_int("Simulation duration (minutes)", 2400)
    DECISION_STEP  = ask_int("Decision step L (minutes)", 1)
    n_runs         = ask_int("Runs per scenario", 10)

    print(f"\\n  -> Config: {NUM_MACHINES} machines | M{BOTTLENECK+1} bottleneck | {SIM_DURATION} min duration | {n_runs} runs/scenario\\n")

    print("Machine Characteristics:")
    for i in range(NUM_MACHINES):
        print(f"  M{i+1}: Cycle={CYCLE_TIME[i]}m, Eta={WEIBULL_ETA[i]}, Beta={WEIBULL_BETA[i]}, P_proc={POWER_PROCESSING[i]}kW")
    print()

    print("Running scenarios...")
    results = {}
    for sc in ('no_decision', 'maint_only', 'energy_only', 'dt_reo'):
        results[sc] = run_average(sc, n_runs=n_runs)

    print_summary(results)
    plot_results(results)

    # ─────────────────────────────────────────────────────────────────────────────
    # SENSITIVITY ANALYSIS (Table 9/10 / Fig 17/18 equivalent)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\nRunning Sensitivity Analysis (Varying Buffer Capacity)...")
    global BUFFER_CAPACITY
    original_buffer = list(BUFFER_CAPACITY)
    
    sens_results = []
    shifts = [-2, -1, 0, 2, 4]
    labels = ["Overall reduction\nof 2 capacities", "Overall reduction\nof 1 capacity", "Original cache", "Overall increase\nof 2 capacities", "Overall increase\nof 4 capacities"]
    
    for i, shift in enumerate(shifts):
        BUFFER_CAPACITY = [max(1, cap + shift) for cap in original_buffer]
        
        # Run 3 times for speed in sensitivity
        sens_runs = max(1, n_runs // 3)
        res_nd = run_average('no_decision', n_runs=sens_runs)
        res_dr = run_average('dt_reo', n_runs=sens_runs)
        
        prof_nd = res_nd['output'] * PRODUCT_VALUE - res_nd['maint_cost'] - res_nd['energy_cost']
        prof_dr = res_dr['output'] * PRODUCT_VALUE - res_dr['maint_cost'] - res_dr['energy_cost']
        
        sens_results.append({
            'Label': labels[i],
            'ND_Energy': res_nd['energy_cost'],
            'DR_Energy': res_dr['energy_cost'],
            'ND_Output': res_nd['output'],
            'DR_Output': res_dr['output'],
            'ND_Profit': prof_nd,
            'DR_Profit': prof_dr,
            'Improvement': (prof_dr - prof_nd) / prof_nd * 100
        })
        
    BUFFER_CAPACITY = original_buffer
    
    print("\nSensitivity Analysis (Cache Capacity vs Total Profit):")
    print("-" * 85)
    print(f"{'Cache Shift':<35} {'No Decision Profit':<20} {'DT-REO Profit':<15} {'Improvement %':>12}")
    print("-" * 85)
    for r in sens_results:
        print(f"{r['Label'].replace('\n', ' '):<35} {r['ND_Profit']:<20.2f} {r['DR_Profit']:<15.2f} {r['Improvement']:>12.2f}%")
    print("-" * 85)

    # Generate Sensitivity Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(labels))
    
    nd_energy = [r['ND_Energy'] for r in sens_results]
    dr_energy = [r['DR_Energy'] for r in sens_results]
    nd_output = [r['ND_Output'] for r in sens_results]
    dr_output = [r['DR_Output'] for r in sens_results]
    
    # Plot 1: Energy Cost
    ax1.plot(x, nd_energy, 'o-', color='#f28e2b', linewidth=2.5, label='No decision', markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax1.plot(x, dr_energy, 'o-', color='#4e79a7', linewidth=2.5, label='DT-REO', markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax1.set_title('Energy Consumption Cost (yuan)', fontweight='bold', fontsize=14, loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', linestyle='-', alpha=0.3)
    for i in range(len(x)):
        ax1.annotate(f"{nd_energy[i]:.2f}", (x[i], nd_energy[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
        ax1.annotate(f"{dr_energy[i]:.2f}", (x[i], dr_energy[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=10)
    
    # Plot 2: Output
    ax2.plot(x, nd_output, 'o-', color='#f28e2b', linewidth=2.5, label='No decision', markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax2.plot(x, dr_output, 'o-', color='#4e79a7', linewidth=2.5, label='DT-REO', markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax2.set_title('Output (piece)', fontweight='bold', fontsize=14, loc='left')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', linestyle='-', alpha=0.3)
    for i in range(len(x)):
        ax2.annotate(f"{nd_output[i]:.2f}", (x[i], nd_output[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=10)
        ax2.annotate(f"{dr_output[i]:.2f}", (x[i], dr_output[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
    
    # Custom legend at the bottom
    handles, lbls = ax1.get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('sensitivity_plots.png', dpi=300, bbox_inches='tight')
    print("Saved 'sensitivity_plots.png'")
    
if __name__ == '__main__':
    main()
