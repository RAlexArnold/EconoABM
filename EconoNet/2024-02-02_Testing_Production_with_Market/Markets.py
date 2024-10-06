# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:18:13 2024

@author: Alex
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:33:15 2024

@author: Alex
"""

import numpy as np
import random
import math
from scipy.special import softmax

class SCDA:
    
    def __init__(self, Nproducts=None, theta=0.1, gamma=1.0, epsilon=0.005, max_tries=100, M_min=0.01):#, peval='log'):
        
        self.Nproducts = Nproducts
        self.theta = theta
        self.epsilon = epsilon
        self.max_tries = max_tries
        self.M_min = M_min
        self.gamma = gamma
        #self.peval = peval
        
        assert self.gamma <= 1.0
        assert self.theta >= 0.0
        
        self.reset()
        

    def reset(self):
        
        self.quantity_exchanged = np.zeros(self.Nproducts)
        self.money_exchanged = np.zeros(self.Nproducts)

        self.quantity_exchanged[:] = 0
        self.money_exchanged[:] = 0
        
        self.n_tries = 0

        self.uncleared_commodities = list(range(self.Nproducts)) 
        
    def determine_market_quantity(self, agent):
        
        S = agent.Q - agent.D
    
        scaling_mask = S > 0.0
        S[scaling_mask] = self.gamma*S[scaling_mask]
    
        return S
    
    def enact_transaction(self, i, buyer_agent, buyer_index, seller_agent, seller_index):
    
        S_buyer     = -self.agent_data[buyer_index, 2, i]
        S_buyer_orig = S_buyer
        buyer_bid    = self.agent_data[buyer_index, 0, i]
        buyer_M      = self.agent_Mdata[buyer_index]
        
        # Check that the buyer has enough money to purchase S_buyer, else 
        if buyer_bid*S_buyer > buyer_M:
            S_buyer = buyer_M/buyer_bid
        
        S_seller     = self.agent_data[seller_index, 2, i]
        S_seller_orig = S_seller
        seller_ask   = self.agent_data[seller_index, 1, i]
        
        # Come to agreement
        # Can't buy more than the seller has, or the buyer wants
        S_exchanged = min(S_buyer, S_seller)
        
        # Select random price between the bidding and asking price.
        #p_exchanged = 10**(random.uniform(math.log10(seller_ask), math.log10(buyer_bid)))
        p_exchanged = math.sqrt(seller_ask*buyer_bid)
        
        dM_exchanged = p_exchanged * S_exchanged
        # Since the bidding price is larger than the asking price, then p_exchanged*S_exchanged will still <= buyer M
    
        # Update the agents' states
        buyer_agent.Q[i]  = buyer_agent.Q[i]  + S_exchanged
        seller_agent.Q[i] = seller_agent.Q[i] - S_exchanged
    
        buyer_agent.M  = buyer_agent.M  - dM_exchanged
        seller_agent.M = seller_agent.M + dM_exchanged
    
    
        # Update the agent_data numpy array
        self.agent_data[buyer_index,  2, i] += S_exchanged
        self.agent_data[seller_index, 2, i] -= S_exchanged
        
        self.agent_Mdata[buyer_index]  = buyer_agent.M
        self.agent_Mdata[seller_index] = seller_agent.M
        
        return p_exchanged, S_exchanged, S_exchanged/S_buyer_orig, S_exchanged/S_seller_orig

    
    def update_prices(self, i):
        
        delta_bidding_price = np.random.uniform(0, [self.theta]*self.Nm)
        delta_asking_price  = np.random.uniform(0, [self.theta]*self.Nm)
        
        # Only those agents which are buying or selling (in the market) will update prices. If an agent is satisfied, they leave market and don't update prices
        # So they must be included in the buyers_mask or sellers_mask
        
        # For buyers
        self.agent_data[self.buyers_mask, 0, i]  *= (1.0 + (1.0 - 2.0*self.buyers_efficiencies[self.buyers_mask])*delta_bidding_price[self.buyers_mask])
        
        # For sellers
        self.agent_data[self.sellers_mask, 1, i] *= (1.0 - (1.0 - 2.0*self.sellers_efficiencies[self.sellers_mask])*delta_asking_price[self.sellers_mask])
        
        # Set an upper limit to the bidding prices, so that agents won't bid more than they have for a unit of a good
        self.agent_data[:, 0, i] = np.minimum(self.agent_data[:, 0, i], self.agent_Mdata)
    
        # Set a lower limit to the asking prices, so that agents don't ask for less than the minimum amount of money required in market (if exists)
        self.agent_data[:, 1, i] = np.maximum(self.M_min, self.agent_data[:, 1, i])
        

    def record_updated_prices(self):
    
        for agent in self.exchanging_agent_list:
    
            agent_index = self.agent_mapping[agent]
    
            new_p_buy  = self.agent_data[agent_index, 0]
            new_p_sell = self.agent_data[agent_index, 1]
    
            agent.p_buy = new_p_buy
            agent.p_sell = new_p_sell 
            
    def attempt_trial(self):
        # For trial, pick random commodity
        i = random.choice(self.uncleared_commodities) # commodity i

        agent_data_i = self.agent_data[:,:,i]
        
        # Create mask for buyers and select their data
        # Eligible buyers have a negative surplus, and have money to spend
        self.buyers_mask = (agent_data_i[:, 2] < 0) & (self.agent_Mdata > self.M_min)
        buyers_data_i = agent_data_i[self.buyers_mask]
    
        # Creat mask for sellers and select their data
        self.sellers_mask = agent_data_i[:, 2] > 0
        sellers_data_i = agent_data_i[self.sellers_mask]
        
        N_buyers  = self.buyers_mask.sum()
        N_sellers = self.sellers_mask.sum()
        
        #self.buyers_transact[:]  = False #  = np.zeros(self.Nm, dtype=bool) # Length of all in transaction, not jus length of buyers
        #self.sellers_transact[:] = False # = np.zeros(self.Nm, dtype=bool)
        self.buyers_efficiencies[:] = 0.0
        self.sellers_efficiencies[:] = 0.0
        
        # If there are no buyers or sellers for commodity i, then this commodity is cleared
        if (N_buyers == 0) or (N_sellers == 0):
            self.uncleared_commodities.remove(i)
            
        else:
            # Continue with commodity
            
            # Select N pairs of buyers and sellers to occur in parallel
            N_transacting_agents = min(N_buyers, N_sellers)
    
            # Using the bidding and asking prices, select a buyer, seller pair for this trial
            # The buyers, sellers are selected with probabilities related to their prices. Use softmax for this
            bidding_prices_i = buyers_data_i[:,0]
            asking_prices_i  = sellers_data_i[:,1]
            
            # Adjust the bids and asks
            adjusted_bids = bidding_prices_i - np.max(bidding_prices_i)
            adjusted_asks = asking_prices_i - np.min(asking_prices_i)
    
            # Normalize the adjusted bids and asks to be in the range [0, 1]
            range_bids = np.max(adjusted_bids) - np.min(adjusted_bids)
            range_asks = np.max(adjusted_asks) - np.min(adjusted_asks)
    
            normalized_bids = (adjusted_bids - np.min(adjusted_bids)) / max(range_bids, 1e-10)  # Avoid division by zero
            normalized_asks = (adjusted_asks - np.min(adjusted_asks)) / max(range_asks, 1e-10)  # Avoid division by zero
    
            # Apply softmax to the normalized values
            softmax_bids = softmax(normalized_bids)
            softmax_asks = softmax(-normalized_asks)            
            
            buyer_agents = np.random.choice(self.exchanging_agent_list[self.buyers_mask],  p=softmax_bids, size=N_transacting_agents, replace=False)
            buyer_indices = [self.agent_mapping[buyer_agent] for buyer_agent in buyer_agents] #agent_mapping[buyer_agents] # will this work?

            seller_agents = np.random.choice(self.exchanging_agent_list[self.sellers_mask],  p=softmax_asks, size=N_transacting_agents, replace=False)
            seller_indices = [self.agent_mapping[seller_agent] for seller_agent in seller_agents] #agent_mapping[seller_agents]
    
            # Check if the selected buyer and seller can make a transaction
            buyer_bids  = agent_data_i[buyer_indices,  0]
            seller_asks = agent_data_i[seller_indices, 1]
    
            ### For each transacting pair, perform a transaction attempt if possible ####
            for n in range(N_transacting_agents):
    
                # Grabbing indices...
                buyer_agent = buyer_agents[n]
                buyer_index = buyer_indices[n]
                buyer_bid = buyer_bids[n]
    
                seller_agent = seller_agents[n]
                seller_index = seller_indices[n]
                seller_ask = seller_asks[n]
    
                # If the bid is higher than the ask, perform the transaction
                if buyer_bid >= seller_ask:
                    
                    p_pair, S_pair, buyer_eff, seller_eff = self.enact_transaction(i, buyer_agent, buyer_index, seller_agent, seller_index)
                    
                    #self.buyers_transact[buyer_index] = True
                    #self.sellers_transact[seller_index] = True
                    #print(buyer_index, seller_index, buyer_bid, seller_ask, p, S_exchanged, buyer_eff, seller_eff)
                    
                    #self.S_exchanged += S_pair
                    #self.M_exchanged += p_pair*S_pair
                    #self.money_exchanged[i]    += p_pair*S_pair
                    #self.quantity_exchanged[i] += S_pair
    
                else:
                    
                    p_pair = 0.0
                    S_pair = 0.0
                    buyer_eff = 0.0
                    seller_eff = 0.0
                    
                    #pass # Nothing to do here, right?
                
                self.buyers_efficiencies[buyer_index] = buyer_eff
                self.sellers_efficiencies[seller_index] = seller_eff
                
                self.money_exchanged[i]    += p_pair*S_pair
                self.quantity_exchanged[i] += S_pair
                
            # After all transacting pairs have succeeded, or failed, we update their prices.
            # Prices are updated depending on if the transaction was successful or not
            # Agents not transacting do not update their prices
            
            # Create masks for the successful and unsuccessful bids/asks
            #self.successful_bids = self.buyers_transact #same as #buyers_mask & buyers_transact
            #self.successful_asks = self.sellers_transact #same as #sellers_mask & sellers_transact
    
            #self.unsuccessful_bids = self.buyers_mask  & ~self.buyers_transact # Those who are buying but don't succeed
            #self.unsuccessful_asks = self.sellers_mask & ~self.sellers_transact # Those who are selling but don't succeed
            
            self.update_prices(i)
            
            self.n_tries += 1
            
    def randomize_prices(self):
    
        for agent in self.exchanging_agent_list:
    
            if random.random() < self.epsilon:
                agent.p_buy = np.random.uniform(0, agent.M, self.Nproducts)
    
            if random.random() < self.epsilon:
                agent.p_sell = np.random.uniform(0, agent.M, self.Nproducts)
    
    
    def run_exchange(self, ExchangeDict):
        
        # self.reset()
        
        self.Nm = len(ExchangeDict) # Number of agents in market
        
        # Create list of agents in market
        self.exchanging_agent_list = np.array([agent for agent in ExchangeDict.keys()])
        self.agent_mapping = {agent_obj: i for i,agent_obj in enumerate(ExchangeDict.keys())}
        
        # Create numpy array of agent data
        self.agent_data = np.array([[agent.p_buy, agent.p_sell, self.determine_market_quantity(agent)] for agent in ExchangeDict.keys()]) # prices and quantity per commodity
        self.agent_Mdata = np.array([agent.M for agent in ExchangeDict.keys()]) # Money
        
        # Record if the buyer/seller has successfully transacted.
        # Reset to False at the beginning of each trial
        #self.buyers_transact  = np.zeros(self.Nm, dtype=bool) # Length of all in transaction, not jus length of buyers
        #self.sellers_transact = np.zeros(self.Nm, dtype=bool)
        
        # Record the quantity efficiency of buyer and sellers transactions.
        # Reset to zero at the beginning of each transaction
        self.buyers_efficiencies = np.zeros(self.Nm)
        self.sellers_efficiencies = np.zeros(self.Nm)
        
        self.reset()
        
        while (len(self.uncleared_commodities) > 0) and (self.n_tries < self.max_tries):
            # Attempt trial to let all buyers/sellers attempt transacting
            self.attempt_trial()
            
        # Save the new bid and asking prices to the agent states
        self.record_updated_prices()
        
        # Randomize bid and asking prices with probability epsilon
        self.randomize_prices()
        
        return self.money_exchanged, self.quantity_exchanged, self.n_tries
           
    
    



















