import time
import random
import numpy as np
from scipy.stats import norm, uniform
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools

import Agents
import Instruments
import Markets

import Simulate
import Environment

class Simulator:
    def __init__(self, Nagents, Nactions, Nproducts, l1, l2, c1, c2, buffer=1.0):
        self.Nagents = Nagents
        self.Nactions = Nactions
        self.Nproducts = Nproducts
        self.l1 = l1
        self.l2 = l2
        self.c1 = c1
        self.c2 = c2
        self.buffer = buffer

        assert c1 <= 1/l1
        c2_max = (1-l1*c1)/l2
        assert c2 <= c2_max

        condition = l1*c1 + l2*c2
        L_nec = condition*Nagents
        print(condition)
        
        self.K1 = int(round(l1*c1/condition*Nagents*buffer, 0))
        self.K2 = int(round(l2*c2/condition*Nagents*buffer, 0))

        self.prod1 = 1./l1
        self.prod2 = 1./l2

        self.initial_parameters = {
            'Q': np.zeros(self.Nproducts),
            'D': np.zeros(self.Nproducts),
            'cg': np.array([c1, c2]),
            'M': 100.,
            'n_actions': self.Nactions
        }

        self.learning_parameters = {
            'gamma': 0.9,
            'epsilon': 1.0,
            'eps_end': 0.05,
            'eps_dec': 0.01,
            'batch_size': 100,
            'input_dims': [3*self.Nproducts+1],
            'lr': 0.005,
            'n_actions': self.Nactions,
        }

        self.market_parameters = {
            'Nproducts': self.Nproducts,
            'theta': 0.1,
            'gamma': 1.0,
            'epsilon': 0.005,
            'max_tries': 100,
            'M_min': 0.01,
        }

        self.env = None
        self.sim = None

    def initialize_agents(self):
        kwargs = {**self.initial_parameters, **self.learning_parameters}
        agent_list = [Agents.Agent(**kwargs) for _ in range(self.Nagents)]
        return agent_list

    def initialize_instruments(self):
        instruments = []
        for N in range(self.K1):
            instrument = Instruments.IoP([self.prod1, 0])
            instruments.append(instrument)

        for N in range(self.K2):
            instrument = Instruments.IoP([0, self.prod2])
            instruments.append(instrument)

        return instruments

    def initialize_allocation(self, agent_list, instrument_list):
        for agent in agent_list:
            random_instrument = random.choice(instrument_list)
            instrument_list.remove(random_instrument)
            agent.Ins = random_instrument

    def initialize_market(self):
        return Markets.SCDA(**self.market_parameters)

    def initialize_environment(self):
        agent_list = self.initialize_agents()
        instrument_list = self.initialize_instruments()
        self.initialize_allocation(agent_list, instrument_list)
        market = self.initialize_market()
        
        # Redefine action space to remove null action
        for agent in agent_list:
            agent.action_space = [1, 2, 3]
        
        return Environment.Environment(1, agent_list, instrument_list, market)

    def initialize_simulator(self, learn_trigger):
        self.env = self.initialize_environment()
        self.sim = Simulate.Simulate(self.env)
        self.sim.set_reward_trigger_learn(learn_trigger)

    def run_simulation(self, T):
        s = time.perf_counter()
        self.sim.run_simulation(T)
        e = time.perf_counter()
        print(f"Simulation took {e-s} seconds")
        
        return e-s

    def calculate_mean_deficits(self):
        log_mean_deficit1 = np.log10(np.mean(self.sim.Darray[:,0], axis=0)).mean() # Take the time-average of each agent's deficit, take log, then find mean of this
        log_mean_deficit2 = np.log10(np.mean(self.sim.Darray[:,1], axis=0)).mean()
        
        return log_mean_deficit1, log_mean_deficit2
    
    def calculate_std_deficits(self):
        log_std_deficit1 = np.log10(np.mean(self.sim.Darray[:,0], axis=0)).std() # Take the time-average of each agent's deficit, take log, then find stdev of this
        log_std_deficit2 = np.log10(np.mean(self.sim.Darray[:,1], axis=0)).std()
        
        return log_std_deficit1, log_std_deficit2

    '''
    def vary_parameters_and_run(self, parameter_dict, parameter_values, Ntimes):
        
        run_times = []
        mean_deficit_1 = []
        mean_deficit_2 = []
        std_deficit_1  = []
        std_deficit_2  = []

        for key, values in parameter_values.items():
            for val in values:
                
                params = {**parameter_dict, key: val}
                
                self.__init__(self.Nagents, self.Nactions, self.Nproducts, self.l1, self.l2, self.c1, self.c2, self.buffer)
                self.learning_parameters.update(params)
                learn_trigger = -self.initial_parameters['cg'].sum() * 10
                
                self.initialize_simulator(learn_trigger)
                run_time = self.run_simulation(Ntimes)
                
                mean_deficits = self.calculate_mean_deficits()
                std_deficits = self.calculate_std_deficits()
                
                run_times.append(run_time)
                
                mean_deficit_1.append((val, mean_deficits[0]))
                mean_deficit_2.append((val, mean_deficits[1]))
                
                std_deficit_1.append((val, std_deficits[0]))
                std_deficit_2.append((val, std_deficits[1]))

        return run_times, mean_deficit_1, mean_deficit_2, std_deficit_1, std_deficit_2
     '''
     
    def vary_parameters_and_run(self, parameter_dict, parameter_values, Ntimes):
    
        # Create an empty list to store DataFrames
        dfs = []
    
        # Generate all combinations of parameter values
        all_combinations = list(itertools.product(*parameter_values.values()))
    
        for combination in all_combinations:
            params = {key: val for key, val in zip(parameter_values.keys(), combination)}
            self.__init__(self.Nagents, self.Nactions, self.Nproducts, self.l1, self.l2, self.c1, self.c2, self.buffer)
    
            # Check if 'cg' is present in initial_parameters
            if 'cg' in self.initial_parameters:
                learn_trigger = -self.initial_parameters['cg'].sum() * 10
            else:
                # Set a default value if 'cg' is not present
                learn_trigger = -10
    
            self.learning_parameters.update(params)
            self.initialize_simulator(learn_trigger)
            
            print(self.learning_parameters)
            run_time = self.run_simulation(Ntimes)
    
            mean_deficits = self.calculate_mean_deficits()
            std_deficits = self.calculate_std_deficits()
    
            # Create a DataFrame for each iteration
            df = pd.DataFrame({
                'Run Time': [run_time],
                'Mean Deficit 1': [mean_deficits[0]],
                'Mean Deficit 2': [mean_deficits[1]],
                'Std Deficit 1': [std_deficits[0]],
                'Std Deficit 2': [std_deficits[1]],
                **params  # Add parameter columns
            })
    
            dfs.append(df)
    
        # Concatenate the list of DataFrames into a single DataFrame
        results_df = pd.concat(dfs, ignore_index=True)
    
        return results_df
        

    
# Example usage:
simulator = Simulator(Nagents=100, Nactions=4, Nproducts=2, l1=1.0, l2=0.5, c1=0.1, c2=0.3)
parameter_dict = {'gamma': 0.9, 'epsilon': 1.0, 'batch_size': 100, 'lr': 0.005}
#parameter_values = {'gamma': [0.8, 0.9, 1.0], 'epsilon': [0.1, 0.2, 0.3], 'batch_size': [50, 100, 150], 'lr': [0.001, 0.005, 0.01]}
parameter_values = {'batch_size': [50, 100, 150], 'lr': [0.001, 0.005, 0.01], 'gamma': [0.8, 0.9, 1.0]}

Ntimes = 1000

results = simulator.vary_parameters_and_run(parameter_dict, parameter_values, Ntimes)

results.to_pickle('parameter_search_2.pkl')