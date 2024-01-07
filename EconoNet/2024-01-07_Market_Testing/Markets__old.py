# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 21:57:36 2023

@author: Alex
"""
import numpy as np
import random

class ProductBank():
    
    def __init__(self, Nproducts, *, alpha = 0.9, max_tries=1000, max_agent_tries = 10):
        
        # Initialize
        self.Nproducts = Nproducts
        self.alpha = alpha
        self.n_tries = 0
        self.max_tries = max_tries
        self.max_agent_tries = max_agent_tries
        self.stock = np.zeros(shape=Nproducts)
        
        self.IntendedExchange = {}
        
        self.total_deposits = np.zeros(shape=Nproducts)
        self.total_withdrawals = np.zeros(shape=Nproducts)
        
        self.tolerance = 1e-6
        
    def reset(self):
        self.n_tries = 0
        self.total_deposits = np.zeros(shape=self.Nproducts)
        self.total_withdrawals = np.zeros(shape=self.Nproducts)
    
    def run_exchange(self, ExchangeDict):
        
        # Create the dictionary of agents and their intended exchanges
        self.IntendedExchange = self.set_intended_exchange(ExchangeDict)
        
        # Initialize the attempts per agent
        self.AgentAttempts = self.set_agent_attempts(ExchangeDict)
        
        
        # Loop while giver and taker sets are non-empty, and the attempts are less than the maximum number (global attempts)
        while (len(self.IntendedExchange) > 0) and (self.n_tries < self.max_tries):
            
            #print(len(self.uncleared_commodities), self.n_tries, self.max_tries)

            self.attempt_exchange()
            
            # The bank-defined (not agent defined) attempts increase by one each round
            self.n_tries += 1
        
        
    def set_intended_exchange(self, ExchangeDict):
        # ExchangeDict : dict - Dictionary of exchaning agents at time t
        
        # Given a dictionary of exchange agents (ExchangeDict) for time t,
        # Initialize the IntendedExchange dictionary
        # This includes {agent : D-Q} for each commodity
        # Where D-Q is the proxy for the intended exchange
        # If D-Q > 0 then agent needs to take this from the Bank (has less than need)
        # If D-Q < 0 then agent can give this to the bank (has more than need)

        IntendedExchange = {}

        for Agent in ExchangeDict:

            E_ = Agent.D - Agent.Q
            
            scaling_mask = E_ < 0
            E_[scaling_mask] *= self.alpha

            IntendedExchange[Agent] = E_

        return IntendedExchange
    
    def set_agent_attempts(self, ExchangeDict):
        
        AgentAttempts = {}
        
        for Agent in ExchangeDict:
            AgentAttempts[Agent] = 0
            
        return AgentAttempts
    
    def attempt_exchange(self):
        
        # Pick a random agent and they either drop off to or take items from the bank
  
        # Pick a random agent to interact with the bank
        agent, agent_E = random.choice(list(self.IntendedExchange.items()))
        
        # For negative values of agent_E, the agent will deposit these into the bank
        deposits = np.where(agent_E < 0, agent_E, 0)
        self.stock = self.stock - deposits
        agent.Q = agent.Q + deposits
        self.total_deposits = self.total_deposits + deposits
        
        # For positive values of agent_E, the agent will take these from the stock, but only as much as the stock allows.
        withdrawals = np.where(agent_E > 0, agent_E, 0)
        # This possible amount taken is min(agent_E_j, stock_j), include a max(0, min(.,.)) to ensure the stock is never below 0. Shouldn't be tough, but what about rounding errors?
        withdrawals = np.maximum(0*self.stock, np.minimum(withdrawals, self.stock))
        self.stock = self.stock - withdrawals
        agent.Q = agent.Q + withdrawals
        self.total_withdrawals = self.total_withdrawals + withdrawals
        
        # Update the IntendedExchange dictionary
        new_E = agent.D - agent.Q
        self.IntendedExchange[agent] = new_E
        
        # Update the agents bank attempts
        self.AgentAttempts[agent] = self.AgentAttempts[agent] + 1
        
        # If the agent is satisfied at the bank, then they will leave
        if np.all(np.abs(new_E) < self.tolerance):
            del self.IntendedExchange[agent]
            del self.AgentAttempts[agent]
            
        # If the agent has reached their max attempts they will leave
        elif self.AgentAttempts[agent] >= self.max_agent_tries:
            del self.IntendedExchange[agent]
            del self.AgentAttempts[agent]
            
class Marketv3():
    
    def __init__(self, Nproducts, *, verbose=False, minimum_price = 0.01, alpha=0.1, beta=0.1, gamma=0.9, epsilon=0.01, max_tries=1000, max_agent_tries=10):
        
        # Initialize
        self.Nproducts = Nproducts
        self.alpha = alpha # Increment prices during failed transactions
        self.beta = beta # Increment prices during successful transactions (alpha <= beta)
        self.gamma = gamma # Intoruduce a softening parameter, gamma, so that selling agents can sell less than their excess (i.e. they save some of their excess for themselves)
        self.epsilon = epsilon # Random chance of switching prices, for price exploration
        
        self.verbose = verbose
        
        self.n_tries = 0
        self.max_tries = max_tries
        self.max_agent_tries = max_agent_tries
        self.tolerance = 1e-6
        self.minimum_price = minimum_price
        
        self.IntendedExchange = {}
        self.AgentAttempts = {}
        
        self.uncleared_commodities = list(range(Nproducts))        
        self.quantity_exchanged = np.zeros(shape=Nproducts)# * np.nan #np.arange(Nproducts)
        self.monetary_exchanged = np.zeros(shape=Nproducts)# * np.nan #np.arange(Nproducts)
        self.price_ssd = np.zeros(shape=Nproducts)
        self.s2 = np.zeros(shape=Nproducts)
    
        self.price = None   
        
    def reset(self):
        
        self.n_tries = 0
        self.monetary_exchanged[:] = 0
        self.quantity_exchanged[:] = 0
        self.price_ssd[:] = 0
        self.s2[:] = 0
        self.uncleared_commodities = list(range(self.Nproducts)) 
        
    def printQ(self, ExchangeDict):
        for agent in ExchangeDict:
            print(agent, agent.Q, agent.D, agent.M)
            
    def printM(self, ExchangeDict):
        for agent in ExchangeDict:
            print(agent, agent.M)
            
    def set_intended_exchange(self, ExchangeDict):
        # ExchangeDict : dict - Dictionary of exchaning agents at time t
        
        # Given a dictionary of exchange agents (ExchangeDict) for time t,
        # Initialize the IntendedExchange dictionary
        # This includes {agent : D-Q} for each commodity
        # Where D-Q is the proxy for the intended exchange
        # If D-Q > 0 then agent needs to buy this from the market (has less than need)
        # If D-Q < 0 then agent can sell this in the market (has more than need)
        
        # Intoruduce a softening parameter, alpha, so that selling agents can sell less than their excess (i.e. they save some of their excess for themselves)

        IntendedExchange = {}

        for Agent in ExchangeDict:

            E_ = np.trunc(Agent.D - Agent.Q)
            
            scaling_mask = E_ < 0
            E_[scaling_mask] *= self.gamma

            IntendedExchange[Agent] = E_

        return IntendedExchange
    
    def set_agent_attempts(self, ExchangeDict):
        
        AgentAttempts = {}
        
        for Agent in ExchangeDict:
            AgentAttempts[Agent] = 0
            
        return AgentAttempts
    
    def price_offer_rule_1(self, agent):
        return random.random()*agent.M
    
    def set_prices_exchange(self, j, buyer, seller):
        
        # Grab the buyers max price they are willing to spend on commodity j
        pmax_buyer = buyer.p_buy[j]
            
        # Grab the sellers min price they are willing to sell commodity j for
        pmin_seller = seller.p_sell[j]
        
        if self.verbose:
            print(f'Buyer Price:   {pmax_buyer:.3f}')
            print(f'Seller Price:  {pmin_seller:.3f}')
            print()
    
        # If the buyer doesn't have enough money, they need to re-adjust their price
        if pmax_buyer > buyer.M:
            pmax_buyer_ = np.random.uniform(0, buyer.M)
            
            if self.verbose:
                print(f'-Buyer price {pmax_buyer:.3f} more than Money Amount {buyer.M:.3f}-')
                print(f'Reset Buyer Price to {pmax_buyer_:.3f}')
                print()
                
            pmax_buyer = pmax_buyer_
            
        # If pmax_buyer < pmin_seller, then either buyer is asking for too little or seller asking for too much
        # Increase buyer price and decrease seller price
        # No sale occurs at this step
        if pmax_buyer < pmin_seller:
            # Adjust buyer's price upwards
            pmax_buyer_  = (1+self.alpha)*pmax_buyer
            # Adjust seller's price downwards
            pmin_seller_ = (1-self.alpha)*pmin_seller
            
            if self.verbose:
                print('-Scenario 1-')
                print('No Exchange')
                print(f'-Buyer Price {pmax_buyer:.3f} lower than Seller Price {pmin_seller:.3f}-')
                print(f'Inc. Buyer Price to {pmax_buyer_:.3f} and dec. Seller Price to {pmin_seller_:.3f}')
                print()
        
        # If pmax_buyer >= pmin_seller, then a selling price can be met.
        # But each will want to do better next time
        # Decrease buyer price and increase seller price
        elif pmax_buyer >= pmin_seller:
            
            if self.verbose:
                print('-Scenario 2-')
                print('Enact Exchange')
                print(f'-Buyer Price {pmax_buyer:.3f} higher than Seller Price {pmin_seller:.3f}-')
                print()
                
            self._exchange(j, pmax_buyer, pmin_seller, buyer, seller)
            
            # After exchange occurs, adjust prices for next round
            # Adjust buyer's price downwards
            pmax_buyer_   = (1-self.beta)*pmax_buyer
            pmax_buyer_   = max(pmax_buyer_, self.minimum_price)
            
            # Adjust seller's price upwards
            pmin_seller_  = (1+self.beta)*pmin_seller
            
            if self.verbose:
                print(f'     Dec. Buyer Price to {pmax_buyer_:.3f} and inc. Seller Price to {pmin_seller_:.3f}')
                print()
        
        # Allow for an exploration of prices for the buyer and seller
        # This may help us escape any ruts.
        # Experiment with where to put this
        # If this doesn't work, then we could do a bigger exploration
        # >> p = U(0,M)
        # Randomly switch buyer price with seller
        # Note that the next buyer/seller price is the current seller/buyer price. Assume the buyer/seller doesn't know what the seller/buyer's price will be, but only knows what they just observed.
        if random.random() < self.epsilon:
            pmax_buyer_ = pmin_seller
            if self.verbose:
                print('Switch Buyer Price to Seller Price {pmax_buyer_:.3f}')
        # Randomly switch seller price with buyer
        if random.random() < self.epsilon:
            pmin_seller_ = pmax_buyer
            if self.verbose:
                print('Switch seller Price to Buyer Price {pmax_seller_:.3f}')
            
        return pmax_buyer_, pmin_seller_
    
    def _exchange(self, j, pmax_buyer, pmin_seller, buyer, seller):
        
        # Select price
        p = np.random.uniform(pmax_buyer, pmin_seller)
        
        if self.verbose:
            print('     -Exchanging-')
            print(f'     Commodity {j}')
            print(f'     pmax_buyer:     {pmax_buyer:.3f}')
            print(f'     pmin_seller:    {pmin_seller}')
            print(f'     Settled Price:  {p}')
            print()
            print(f'     Buyer Money:    {buyer.M:.3f}')
            print(f'     Buyer Deficit:  {buyer.D - buyer.Q}')
            print()
            print(f'     Seller Money:   {seller.M:.3f}')
            print(f'     Seller Deficit: {seller.D - seller.Q}')
            print()
            print(f'     -Swap Items {j}-')
            
            
        # Include the quantity and money exchanged for commodity j for this transaction
        _n = self.quantity_exchanged[j]
        _P = self.monetary_exchanged[j]
        _m = _P/_n
        
        n = _n + 1
        P = _P + p
        m = P/n
                     
        self.quantity_exchanged[j] = n
        self.monetary_exchanged[j] = P
        
        self.price_ssd[j] = self.price_ssd[j] + (p - _m)*(p - m)
        self.s2[j] = self.price_ssd[j]/(n)
        
        if self.verbose:
            print(f'     Market M {j}:  {P:.3f}')
            print(f'     Market Q {j}:  {n}')
            print(f'     Market p {j}:  {P/n:.3f}')
            print()

        
        buyer.M = buyer.M - p
        seller.M = seller.M + p
        
        buyer.Q[j] = buyer.Q[j] + 1
        seller.Q[j] = seller.Q[j] - 1
        
        if self.verbose:
           print(f'     Buyer Money:    {buyer.M:.3f}')
           print(f'     Buyer Deficit:  {buyer.D - buyer.Q}')
           print()
           print(f'     Seller Money:   {seller.M}')
           print(f'     Seller Deficit: {seller.D - seller.Q}')
           print('                ------------                  ')
           print()
        
        # Update the IntendedExchange dictionary for the relevant agents
        self.IntendedExchange[buyer] = buyer.D - buyer.Q
        self.IntendedExchange[seller] = seller.D - seller.Q
        
        # Update the agents bank attempts DONE IN OUTSIDE FUNCTION
        #self.AgentAttempts[buyer] = self.AgentAttempts[buyer] + 1
        #self.AgentAttempts[seller] = self.AgentAttempts[seller] + 1

    def attempt_exchange(self):
        
        # Pick a random uncleared commodity
        # Select random buyer and seller
        # Attempt offer price, and market exchange
        # Perform exchange if possible

        # Pick a random uncleared commodity
        j = random.choice(self.uncleared_commodities)

        # For commodity j create the set of buyers and sellers
        # Buyers have a positive intended exchange (want to increase their commodity j)
        # Sellers have a negative intended exchange (want to decrease their commodity j)
        BuyerSet = {n:E[j] for (n,E) in self.IntendedExchange.items() if E[j] > 0}
        SellerSet = {n:E[j] for (n,E) in self.IntendedExchange.items() if E[j] < 0}
        
        if self.verbose:
            print('--Attempt An Exchange--')
            print(j)
            #print(f'Buyers {BuyerSet}')
            #print(f'Sellers {SellerSet}')
            print()

        # If BuyerSet or SellerSet are empty, then remove j from market
        # OR in this timestep there is only 1 market iteration for each agent
        # OR we define a 'speed' v of market, so that there are max ~v*dt market iterations
        # Regardless, if Buyer or Seller Set are empty, then j is removed from market
        if (len(BuyerSet) == 0) or (len(SellerSet) == 0):
            self.uncleared_commodities.remove(j)
            
            if self.verbose:
                print(f'Commodity {j} cleared')
                print()
            
        else:

            # Select random buyer
            buyer, buyer_E = random.choice(list(BuyerSet.items()))

            # Select random seller
            seller, seller_E = random.choice(list(SellerSet.items()))
            
            if self.verbose:
                print(f'Buyer:     {buyer}')
                print(f'Buyer E:   {buyer_E:.3f}')
                print(f'Seller:    {seller}')
                print(f'Seller E:  {seller_E:.3f}')
                print()
                print('Set Prices and Try an Exchange...')
                print()

            # Buyes and sellers will exchange one item at a time
            
            # if p are none, then make prices random
            
            ############################################################################
            # Now we can make some comparisons to enact the sale and enact exchange
            ##
            pmax_buyer_, pmin_seller_ = self.set_prices_exchange(j, buyer, seller)
            ##
            ############################################################################


            # Regardless of exchange or no, update the agent's prices
            buyer.p_buy[j] = max(pmax_buyer_, self.minimum_price)
            seller.p_sell[j] = max(pmin_seller_, self.minimum_price)
            
            if self.verbose:
                print('...After Trying Exchange, Set New Agent Prices')
                print(f'New Buyer Price:     {buyer.p_buy[j]:.3f}')
                print(f'New Seller Price:    {seller.p_sell[j]:.3f}')
                print()
            # Exits regardless of if successful. Any attempt is counted toward max tries when code is placed here.
            # The exit conditions for each agent
            # The exit conditions for each agent
            for agent in [buyer, seller]:
                
                # Increment each attempt
                self.AgentAttempts[agent] = self.AgentAttempts[agent] + 1
                if self.verbose:
                    print(f'Agent {agent} attempts increased from {self.AgentAttempts[agent]-1} to {self.AgentAttempts[agent]}')
                
                # If the agent is satisfied at the market, then they will leave #(does this work for both selelrs and buyers?)
                if np.all(np.abs(self.IntendedExchange[agent]) < 1.0 ): #self.tolerance):
                    
                    if self.verbose:
                        print(f'Agent {agent} Satisified. Exit')
                        print()
                    del self.IntendedExchange[agent]
                    del self.AgentAttempts[agent]
                    
                    
                # If the agent has reached their max attempts they will leave
                elif self.AgentAttempts[agent] >= self.max_agent_tries:
                    if self.verbose:
                        print(f'Agent {agent} has had {self.AgentAttempts[agent]}/{self.max_agent_tries} attempts. Tap Out. Exit')
                        print()
                    del self.IntendedExchange[agent]
                    del self.AgentAttempts[agent]
            
            if self.verbose:
                print('            ------------                      ')
                    

    def run_exchange(self, ExchangeDict):
        
        # Reset the price info (use modified version of reset)
        self.reset()
        
        self.IntendedExchange = self.set_intended_exchange(ExchangeDict)
        self.AgentAttempts = self.set_agent_attempts(ExchangeDict)
        
        #print(self.IntendedExchange)
        
        if self.verbose:
            print('---Begin Exchanging---')
            print()

        # Loop while buyer and seller non empty (len[uncleared_commodites] > 0) or if n_tries > max_tries
        while (len(self.uncleared_commodities) > 0) and (self.n_tries < self.max_tries):
            
            
            #display(f'Commodities {self.uncleared_commodities}')
            #print(self.n_tries, self.max_tries)
            self.attempt_exchange()
            self.n_tries += 1
            if self.verbose:
                print(f'Total Exchange Attempts {self.n_tries}/{self.max_tries}')
                print()
            
        if self.verbose:
            print('            ------------                      ')
            
        avg_price = self.monetary_exchanged/self.quantity_exchanged
        std_price = self.s2
        
        #print()
        
        return self.monetary_exchanged, self.quantity_exchanged, self.s2, self.n_tries
 
        
        
class Marketv4():
    
    def __init__(self, Nproducts, *, verbose=False, minimum_price = 0.01, alpha=0.1, beta=0.1, gamma=0.9, epsilon=0.01, max_tries=1000, max_agent_tries=10):
        
        # Initialize
        self.Nproducts = Nproducts
        self.alpha = alpha # Increment prices during failed transactions
        self.beta = beta # Increment prices during successful transactions (alpha <= beta)
        self.gamma = gamma # Intoruduce a softening parameter, gamma, so that selling agents can sell less than their excess (i.e. they save some of their excess for themselves)
        self.epsilon = epsilon # Random chance of switching prices, for price exploration
        
        self.verbose = verbose
        
        self.n_tries = 0
        self.max_tries = max_tries
        self.max_agent_tries = max_agent_tries
        self.tolerance = 1e-6
        self.minimum_price = minimum_price
        
        self.IntendedExchange = {}
        self.AgentAttempts = {}
        
        self.uncleared_commodities = list(range(Nproducts))        
        self.quantity_exchanged = np.zeros(shape=Nproducts)# * np.nan #np.arange(Nproducts)
        self.monetary_exchanged = np.zeros(shape=Nproducts)# * np.nan #np.arange(Nproducts)
        self.price_ssd = np.zeros(shape=Nproducts)
        self.s2 = np.zeros(shape=Nproducts)
    
        self.price = None   
        
    def reset(self):
        
        self.n_tries = 0
        self.monetary_exchanged[:] = 0
        self.quantity_exchanged[:] = 0
        self.price_ssd[:] = 0
        self.s2[:] = 0
        self.uncleared_commodities = list(range(self.Nproducts)) 
        
    def printQ(self, ExchangeDict):
        for agent in ExchangeDict:
            print(agent, agent.Q, agent.D, agent.M)
            
    def printM(self, ExchangeDict):
        for agent in ExchangeDict:
            print(agent, agent.M)
            
    def set_intended_exchange(self, ExchangeDict):
        # ExchangeDict : dict - Dictionary of exchaning agents at time t
        
        # Given a dictionary of exchange agents (ExchangeDict) for time t,
        # Initialize the IntendedExchange dictionary
        # This includes {agent : D-Q} for each commodity
        # Where D-Q is the proxy for the intended exchange
        # If D-Q > 0 then agent needs to buy this from the market (has less than need)
        # If D-Q < 0 then agent can sell this in the market (has more than need)
        
        # Intoruduce a softening parameter, alpha, so that selling agents can sell less than their excess (i.e. they save some of their excess for themselves)

        IntendedExchange = {}

        for Agent in ExchangeDict:

            E_ = np.trunc(Agent.D - Agent.Q)
            
            scaling_mask = E_ < 0
            E_[scaling_mask] *= self.gamma

            IntendedExchange[Agent] = E_

        return IntendedExchange
    
    def set_agent_attempts(self, ExchangeDict):
        
        AgentAttempts = {}
        
        for Agent in ExchangeDict:
            AgentAttempts[Agent] = 0
            
        return AgentAttempts
    
    def price_offer_rule_1(self, agent):
        return random.random()*agent.M
    
    def set_prices_exchange(self, j, buyer, seller):
        
        # Grab the buyers max price they are willing to spend on commodity j
        pmax_buyer = buyer.p_buy[j]
            
        # Grab the sellers min price they are willing to sell commodity j for
        pmin_seller = seller.p_sell[j]
        
        if self.verbose:
            print(f'Buyer Price:   {pmax_buyer:.3f}')
            print(f'Seller Price:  {pmin_seller:.3f}')
            print()
    
        # If the buyer doesn't have enough money, they need to re-adjust their price
        if pmax_buyer > buyer.M:
            pmax_buyer_ = np.random.uniform(0, buyer.M)
            
            if self.verbose:
                print(f'-Buyer price {pmax_buyer:.3f} more than Money Amount {buyer.M:.3f}-')
                print(f'Reset Buyer Price to {pmax_buyer_:.3f}')
                print()
                
            pmax_buyer = pmax_buyer_
            
        # If pmax_buyer < pmin_seller, then either buyer is asking for too little or seller asking for too much
        # Increase buyer price and decrease seller price
        # No sale occurs at this step
        if pmax_buyer < pmin_seller:
            # Adjust buyer's price upwards
            pmax_buyer_  = pmax_buyer+self.alpha
            # Adjust seller's price downwards
            pmin_seller_ = pmin_seller-self.alpha
            
            if self.verbose:
                print('-Scenario 1-')
                print('No Exchange')
                print(f'-Buyer Price {pmax_buyer:.3f} lower than Seller Price {pmin_seller:.3f}-')
                print(f'Inc. Buyer Price to {pmax_buyer_:.3f} and dec. Seller Price to {pmin_seller_:.3f}')
                print()
        
        # If pmax_buyer >= pmin_seller, then a selling price can be met.
        # But each will want to do better next time
        # Decrease buyer price and increase seller price
        elif pmax_buyer >= pmin_seller:
            
            if self.verbose:
                print('-Scenario 2-')
                print('Enact Exchange')
                print(f'-Buyer Price {pmax_buyer:.3f} higher than Seller Price {pmin_seller:.3f}-')
                print()
                
            self._exchange(j, pmax_buyer, pmin_seller, buyer, seller)
            
            # After exchange occurs, adjust prices for next round
            # Adjust buyer's price downwards
            pmax_buyer_   = pmax_buyer-self.beta
            pmax_buyer_   = max(pmax_buyer_, self.minimum_price)
            
            # Adjust seller's price upwards
            pmin_seller_  = pmin_seller+self.beta
            
            if self.verbose:
                print(f'     Dec. Buyer Price to {pmax_buyer_:.3f} and inc. Seller Price to {pmin_seller_:.3f}')
                print()
        
        # Allow for an exploration of prices for the buyer and seller
        # This may help us escape any ruts.
        # Experiment with where to put this
        # If this doesn't work, then we could do a bigger exploration
        # >> p = U(0,M)
        # Randomly switch buyer price with seller
        # Note that the next buyer/seller price is the current seller/buyer price. Assume the buyer/seller doesn't know what the seller/buyer's price will be, but only knows what they just observed.
        if random.random() < self.epsilon:
            pmax_buyer_ = pmin_seller
            if self.verbose:
                print('Switch Buyer Price to Seller Price {pmax_buyer_:.3f}')
        # Randomly switch seller price with buyer
        if random.random() < self.epsilon:
            pmin_seller_ = pmax_buyer
            if self.verbose:
                print('Switch seller Price to Buyer Price {pmax_seller_:.3f}')
            
        return pmax_buyer_, pmin_seller_
    
    def _exchange(self, j, pmax_buyer, pmin_seller, buyer, seller):
        
        # Select price
        p = np.random.uniform(pmax_buyer, pmin_seller)
        
        if self.verbose:
            print('     -Exchanging-')
            print(f'     Commodity {j}')
            print(f'     pmax_buyer:     {pmax_buyer:.3f}')
            print(f'     pmin_seller:    {pmin_seller}')
            print(f'     Settled Price:  {p}')
            print()
            print(f'     Buyer Money:    {buyer.M:.3f}')
            print(f'     Buyer Deficit:  {buyer.D - buyer.Q}')
            print()
            print(f'     Seller Money:   {seller.M:.3f}')
            print(f'     Seller Deficit: {seller.D - seller.Q}')
            print()
            print(f'     -Swap Items {j}-')
            
            
        # Include the quantity and money exchanged for commodity j for this transaction
        _n = self.quantity_exchanged[j]
        _P = self.monetary_exchanged[j]
        _m = _P/_n
        
        n = _n + 1
        P = _P + p
        m = P/n
                     
        self.quantity_exchanged[j] = n
        self.monetary_exchanged[j] = P
        
        self.price_ssd[j] = self.price_ssd[j] + (p - _m)*(p - m)
        self.s2[j] = self.price_ssd[j]/(n)
        
        if self.verbose:
            print(f'     Market M {j}:  {P:.3f}')
            print(f'     Market Q {j}:  {n}')
            print(f'     Market p {j}:  {P/n:.3f}')
            print()

        
        buyer.M = buyer.M - p
        seller.M = seller.M + p
        
        buyer.Q[j] = buyer.Q[j] + 1
        seller.Q[j] = seller.Q[j] - 1
        
        if self.verbose:
           print(f'     Buyer Money:    {buyer.M:.3f}')
           print(f'     Buyer Deficit:  {buyer.D - buyer.Q}')
           print()
           print(f'     Seller Money:   {seller.M}')
           print(f'     Seller Deficit: {seller.D - seller.Q}')
           print('                ------------                  ')
           print()
        
        # Update the IntendedExchange dictionary for the relevant agents
        self.IntendedExchange[buyer] = buyer.D - buyer.Q
        self.IntendedExchange[seller] = seller.D - seller.Q
        
        # Update the agents bank attempts DONE IN OUTSIDE FUNCTION
        #self.AgentAttempts[buyer] = self.AgentAttempts[buyer] + 1
        #self.AgentAttempts[seller] = self.AgentAttempts[seller] + 1

    def attempt_exchange(self):
        
        # Pick a random uncleared commodity
        # Select random buyer and seller
        # Attempt offer price, and market exchange
        # Perform exchange if possible

        # Pick a random uncleared commodity
        j = random.choice(self.uncleared_commodities)

        # For commodity j create the set of buyers and sellers
        # Buyers have a positive intended exchange (want to increase their commodity j)
        # Sellers have a negative intended exchange (want to decrease their commodity j)
        BuyerSet = {n:E[j] for (n,E) in self.IntendedExchange.items() if E[j] > 0}
        SellerSet = {n:E[j] for (n,E) in self.IntendedExchange.items() if E[j] < 0}
        
        if self.verbose:
            print('--Attempt An Exchange--')
            print(j)
            #print(f'Buyers {BuyerSet}')
            #print(f'Sellers {SellerSet}')
            print()

        # If BuyerSet or SellerSet are empty, then remove j from market
        # OR in this timestep there is only 1 market iteration for each agent
        # OR we define a 'speed' v of market, so that there are max ~v*dt market iterations
        # Regardless, if Buyer or Seller Set are empty, then j is removed from market
        if (len(BuyerSet) == 0) or (len(SellerSet) == 0):
            self.uncleared_commodities.remove(j)
            
            if self.verbose:
                print(f'Commodity {j} cleared')
                print()
            
        else:

            # Select random buyer
            buyer, buyer_E = random.choice(list(BuyerSet.items()))

            # Select random seller
            seller, seller_E = random.choice(list(SellerSet.items()))
            
            if self.verbose:
                print(f'Buyer:     {buyer}')
                print(f'Buyer E:   {buyer_E:.3f}')
                print(f'Seller:    {seller}')
                print(f'Seller E:  {seller_E:.3f}')
                print()
                print('Set Prices and Try an Exchange...')
                print()

            # Buyes and sellers will exchange one item at a time
            
            # if p are none, then make prices random
            
            ############################################################################
            # Now we can make some comparisons to enact the sale and enact exchange
            ##
            pmax_buyer_, pmin_seller_ = self.set_prices_exchange(j, buyer, seller)
            ##
            ############################################################################


            # Regardless of exchange or no, update the agent's prices
            buyer.p_buy[j] = max(pmax_buyer_, self.minimum_price)
            seller.p_sell[j] = max(pmin_seller_, self.minimum_price)
            
            if self.verbose:
                print('...After Trying Exchange, Set New Agent Prices')
                print(f'New Buyer Price:     {buyer.p_buy[j]:.3f}')
                print(f'New Seller Price:    {seller.p_sell[j]:.3f}')
                print()
            # Exits regardless of if successful. Any attempt is counted toward max tries when code is placed here.
            # The exit conditions for each agent
            # The exit conditions for each agent
            for agent in [buyer, seller]:
                
                # Increment each attempt
                self.AgentAttempts[agent] = self.AgentAttempts[agent] + 1
                if self.verbose:
                    print(f'Agent {agent} attempts increased from {self.AgentAttempts[agent]-1} to {self.AgentAttempts[agent]}')
                
                # If the agent is satisfied at the market, then they will leave #(does this work for both selelrs and buyers?)
                if np.all(np.abs(self.IntendedExchange[agent]) < 1.0 ): #self.tolerance):
                    
                    if self.verbose:
                        print(f'Agent {agent} Satisified. Exit')
                        print()
                    del self.IntendedExchange[agent]
                    del self.AgentAttempts[agent]
                    
                    
                # If the agent has reached their max attempts they will leave
                elif self.AgentAttempts[agent] >= self.max_agent_tries:
                    if self.verbose:
                        print(f'Agent {agent} has had {self.AgentAttempts[agent]}/{self.max_agent_tries} attempts. Tap Out. Exit')
                        print()
                    del self.IntendedExchange[agent]
                    del self.AgentAttempts[agent]
            
            if self.verbose:
                print('            ------------                      ')
                    

    def run_exchange(self, ExchangeDict):
        
        # Reset the price info (use modified version of reset)
        self.reset()
        
        self.IntendedExchange = self.set_intended_exchange(ExchangeDict)
        self.AgentAttempts = self.set_agent_attempts(ExchangeDict)
        
        #print(self.IntendedExchange)
        
        if self.verbose:
            print('---Begin Exchanging---')
            print()

        # Loop while buyer and seller non empty (len[uncleared_commodites] > 0) or if n_tries > max_tries
        while (len(self.uncleared_commodities) > 0) and (self.n_tries < self.max_tries):
            
            
            #display(f'Commodities {self.uncleared_commodities}')
            #print(self.n_tries, self.max_tries)
            self.attempt_exchange()
            self.n_tries += 1
            if self.verbose:
                print(f'Total Exchange Attempts {self.n_tries}/{self.max_tries}')
                print()
            
        if self.verbose:
            print('            ------------                      ')
            
        avg_price = self.monetary_exchanged/self.quantity_exchanged
        std_price = self.s2
        
        #print()
        
        return self.monetary_exchanged, self.quantity_exchanged, self.s2, self.n_tries                  
        
    