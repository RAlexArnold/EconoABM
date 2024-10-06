# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 02:01:51 2024

@author: Alex
"""

import numpy as np
import scipy
import random
import math

class SCDA:
    
    def __init__(self, Nproducts, theta=0.1, gamma=1.0, epsilon=0.01, max_tries=100, M_min=0.01, peval='log'):
        
        self.Nproducts = Nproducts
        self.theta = theta
        self.epsilon = epsilon
        self.max_tries = max_tries
        self.M_min = M_min
        self.gamma = gamma
        self.peval = peval
        
        assert self.gamma <= 1.0
        assert self.theta >= 0.0
        
        self.uncleared_commodities = list(range(Nproducts))
        self.quantity_exchanged = np.zeros(Nproducts)
        self.money_exchanged = np.zeros(Nproducts)
        
    def determine_market_quantity(self, agent):
        
        S = agent.Q - agent.D
    
        scaling_mask = S > 0.0
        S[scaling_mask] = self.gamma*S[scaling_mask]
    
        return S
    
    def enact_transaction(self, i, buyer_agent, buyer_index, seller_agent, seller_index):
    
        S_buyer   = -self.agent_data[buyer_index, 2, i]
        buyer_bid  = self.agent_data[buyer_index, 0, i]
        buyer_M    = self.agent_Mdata[buyer_index]
        
        # Check that the buyer has enough money to purchase S_buyer, else decrease the quantity buyer to purchase
        if buyer_bid*S_buyer > buyer_M:
            S_buyer = buyer_M/buyer_bid
        
        S_seller   = self.agent_data[seller_index, 2, i]
        seller_ask = self.agent_data[seller_index, 1, i]
        
        # Come to agreement
        # Can't buy more than the seller has, or the buyer wants
        S_exchanged = min(S_buyer, S_seller)
        # Select random price between the bidding and asking price.
        
        '''
        if self.peval=='linear_rand':   
            p_exchanged = random.uniform(seller_ask, buyer_bid)
        elif self.peval=='log_rand':
            p_exchanged = 10**(random.uniform(math.log10(seller_ask), math.log10(buyer_bid)))
        elif self.peval=='linear_mean':
            p_exchanged = 0.5*(seller_ask + buyer_bid)
        elif self.peval=='log_mean':
            p_exchanged = math.sqrt(seller_ask*buyer_bid)
        else:
            raise ValueError(f"peval is {self.peval}. Should be 'linear_mean', 'log_mean', 'linear_rand', or 'log_rand'")
        '''
            
        p_exchanged = math.sqrt(seller_ask*buyer_bid)
            
        dM_exchanged = p_exchanged * S_exchanged
        # Since the bidding price is larger than the asking price, then p_exchanged*S_exchanged will still <= buyer M
    
        # Update the agents' states
        buyer_agent.Q[i]  = buyer_agent.Q[i]  + S_exchanged
        seller_agent.Q[i] = seller_agent.Q[i] - S_exchanged
    
        buyer_agent.M     = buyer_agent.M  - dM_exchanged
        seller_agent.M    = seller_agent.M + dM_exchanged
    
        # Update the agent_data numpy array
        self.agent_data[buyer_index,  2, i] += S_exchanged
        self.agent_data[seller_index, 2, i] -= S_exchanged
        
        self.agent_Mdata[buyer_index]  = buyer_agent.M
        self.agent_Mdata[seller_index] = seller_agent.M
        
        return p_exchanged, S_exchanged#, S_exchanged/S_buyer, S_exchanged/S_seller # S_buyer will be adjusted if they don't have the money
    
    def update_prices(self, i, buyer_index, seller_index, transaction_successful):
    
        delta_bidding_price = np.random.uniform(0, [self.theta]*self.Nm)
        delta_asking_price  = np.random.uniform(0, [self.theta]*self.Nm)    
    
        if transaction_successful:
    
            #### Buyers ####
            # The selected buyer will decrease their bid for next time
            self.agent_data[buyer_index, 0, i] *= (1.0 - delta_bidding_price[buyer_index])
    
            # The non selected buyers will increase their bids
            non_selected_buyers_mask = self.buyers_mask
            non_selected_buyers_mask[buyer_index] = False # This will update the buyers_mask array
            self.agent_data[non_selected_buyers_mask, 0, i] *= (1.0 + delta_bidding_price[non_selected_buyers_mask])
    
            #### Sellers ####
            # The selected seller will increase their ask for next time
            self.agent_data[seller_index, 1, i] *= (1.0 + delta_asking_price[seller_index])
    
            # The non selected sellers will decrease their asks
            non_selected_sellers_mask = self.sellers_mask
            non_selected_sellers_mask[seller_index] = False # This will update the sellers_mask array
            self.agent_data[non_selected_sellers_mask, 1, i] *= (1.0 - delta_asking_price[non_selected_sellers_mask])
    
        else:
    
            #### Buyers ####
            # No transaction occurs, so all buyers will increase their bidding prices
            self.agent_data[self.buyers_mask, 0, i] *= (1.0 + delta_bidding_price[self.buyers_mask])
    
            #### Sellers ####
            # No transaction occurs, so all selelrs will decrease their asking prices
            self.agent_data[self.sellers_mask, 1, i] *= (1.0 - delta_asking_price[self.sellers_mask])
            
        # Set an upper limit to the bidding prices, so that agents won't bid more than they have for a unit of a good
        self.agent_data[:, 0, i] = np.minimum(self.agent_data[:, 0, i], self.agent_Mdata)
        
        # Set a lower limit on the asking prices, this is equal to the smallest amount of money meaningful in the market (self.M_min)
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
        
        # If there are no buyers or sellers for commodity i, then this commodity is cleared
        if (self.buyers_mask.sum() == 0) or (self.sellers_mask.sum() == 0):
            self.uncleared_commodities.remove(i)
            
        else:
            # Continue with commodity
    
            # Using the bidding and asking prices, select a buyer, seller pair for this trial
            # The buyers, sellers are selected with probabilities related to their prices. Use softmax for this
            bidding_prices_i = buyers_data_i[:,0]
            asking_prices_i  = sellers_data_i[:,1]
            softmax_bids = scipy.special.softmax(bidding_prices_i) #scipy function avoids overflow errors
            softmax_asks = scipy.special.softmax(-asking_prices_i)
    
            buyer_agent  = np.random.choice(self.exchanging_agent_list[self.buyers_mask],  p=softmax_bids)
            buyer_index = self.agent_mapping[buyer_agent]
    
            seller_agent = np.random.choice(self.exchanging_agent_list[self.sellers_mask], p=softmax_asks)
            seller_index = self.agent_mapping[seller_agent]
    
            # Check if the selected buyer and seller can make a transaction
            buyer_bid  = agent_data_i[buyer_index,  0]
            seller_ask = agent_data_i[seller_index, 1]
    
            if buyer_bid >= seller_ask:
                # Transaction
                p, S_exchanged = self.enact_transaction(i, buyer_agent, buyer_index, seller_agent, seller_index)
                transaction_successful = True
    
            else:
                transaction_successful = False
                p, S_exchanged = 0., 0.
    
            self.update_prices(i, buyer_index, seller_index, transaction_successful)
            
            self.money_exchanged[i] += p*S_exchanged
            self.quantity_exchanged[i] += S_exchanged
                
            self.n_tries += 1
                    
    def run_exchange(self, ExchangeDict):
        
        # self.reset()
        
        self.Nm = len(ExchangeDict) # Number of agents in market
        
        # Create list of agents in market
        self.exchanging_agent_list = np.array([agent for agent in ExchangeDict.keys()])
        self.agent_mapping = {agent_obj: i for i,agent_obj in enumerate(ExchangeDict.keys())}
        
        # Create numpy array of agent data
        self.agent_data = np.array([[agent.p_buy, agent.p_sell, self.determine_market_quantity(agent)] for agent in ExchangeDict.keys()]) # prices and quantity per commodity
        self.agent_Mdata = np.array([agent.M for agent in ExchangeDict.keys()]) # Money
        
        self.n_tries = 0
        self.money_exchanged[:] = 0
        self.quantity_exchanged[:] = 0
        self.uncleared_commodities = list(range(self.Nproducts)) 
        
        
        while (len(self.uncleared_commodities) > 0) and (self.n_tries < self.max_tries):
            self.attempt_trial()
            
        # Save the new bid and asking prices to the agent states
        self.record_updated_prices()
        
        return self.money_exchanged, self.quantity_exchanged, self.n_tries
        
'''
def randomize_agent_bid(agent, i):
    
    #new_bid = np.random.uniform(M_min, agent.M)
    new_bid = random_bid_list[eps_agent_list < epsilon]
    agent.p_buy[i] = new_bid
    
def randomize_agent_ask(agent, i):
    
    new_ask = np.random.uniform(M_min, agent.M)
    agent.p_sell[i] = new_ask
    
epsilon = 1e-2
eps_agent_list = np.random.uniform(0,1, Nm)

    
bid_dist_i = agent_data[:, 0, i]

mean_bid = bid_dist_i.mean()
std_bid = bid_dist_i.std()

random_bid_list = std_bid/2.*np.random.standard_cauchy(Nm) + mean_bid

[randomize_agent_bid(agent, i) for agent in exchanging_agent_list[eps_agent_list < epsilon]]
'''
                
        
        