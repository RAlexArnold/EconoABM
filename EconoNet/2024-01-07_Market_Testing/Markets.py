# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 23:58:09 2023

@author: Alex
"""

import numpy as np
import random

class Market():
    
    def __init__(self, Nproducts, *, verbose=False, minimum_price = 0.01, alpha=0.1, beta=0.1, gamma=0.9, epsilon_switch=0.01, epsilon_rand=0.01, max_tries=1000, max_agent_tries=10, price_adjust_method='step'):
        
        # Initialize
        self.Nproducts = Nproducts
        self.alpha = alpha # Increment prices during failed transactions
        self.beta = beta # Increment prices during successful transactions (alpha <= beta)
        self.gamma = gamma # Intoruduce a softening parameter, gamma, so that selling agents can sell less than their excess (i.e. they save some of their excess for themselves)
        self.epsilon_switch = epsilon_switch # Random chance of switching prices, for price exploration
        self.epsilon_rand = epsilon_rand # Random reassignment of prices for price exploration
        
        self.price_adjust_method = price_adjust_method
        
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

            #E_ = np.trunc(Agent.D - Agent.Q)
            
            E_ = Agent.D - Agent.Q
            
            scaling_mask = E_ < 0
            E_[scaling_mask] = self.gamma*E_[scaling_mask] #*= self.gamma
            
            E_ = np.trunc(Agent.D - Agent.Q)

            IntendedExchange[Agent] = E_

        return IntendedExchange
    
    def set_agent_attempts(self, ExchangeDict):
        
        AgentAttempts = {}
        
        for Agent in ExchangeDict:
            AgentAttempts[Agent] = 0
            
        return AgentAttempts
    
    def price_offer_rule_1(self, agent):
        return random.random()*agent.M
    
    def adjust_price(self, p, theta, *, method='step'):
        
        if method=='proportional':
            return (1.0 + theta)*p
        
        elif method=='step':
            return (p + theta)
        
        elif method=='uniform':
            return np.random.uniform(0, theta)
    
    def set_prices_exchange(self, j, buyer, seller):
        
        # Grab the buyers max price they are willing to spend on commodity j
        pmax_buyer = buyer.p_buy[j]
            
        # Grab the sellers min price they are willing to sell commodity j for
        pmin_seller = seller.p_sell[j]
        
        if self.verbose:
            print('<<set_prices_exchange()>>')
            print(f'Buyer {buyer}')
            print(f'Seller {seller}')
            print(f'Buyer Price:   {pmax_buyer:.3f}')
            print(f'Seller Price:  {pmin_seller:.3f}')
            print()
    
        # If the buyer doesn't have enough money, they need to re-adjust their price
        if pmax_buyer > buyer.M:
            pmax_buyer_ = self.adjust_price(0, buyer.M, method='uniform') # np.random.uniform(0, buyer.M)
            
            if self.verbose:
                print(f'-Buyer price {pmax_buyer:.3f} more than Money Amount {buyer.M:.3f}-')
                print(f'Reset Buyer Price to {pmax_buyer_:.3f}')
                print()
                
            pmax_buyer = pmax_buyer_
            
        # If pmax_buyer < pmin_seller, then either buyer is asking for too little or seller asking for too much
        # Increase buyer price and decrease seller price
        # No sale occurs at this step
        if pmax_buyer < pmin_seller:
            
            
            if (self.price_adjust_method == 'step') or (self.price_adjust_method == 'proportional'):
                # Adjust buyer's price upwards
                pmax_buyer_  = self.adjust_price(pmax_buyer, self.alpha) #(1+self.alpha)*pmax_buyer
                # Adjust seller's price downwards
                pmin_seller_ = self.adjust_price(pmin_seller, -self.alpha) #(1-self.alpha)*pmin_seller
                
            elif (self.price_adjust_method == 'uniform'):
                pmax_buyer_ = self.adjust_price(0, buyer.M)
                pmin_seller_ = self.adjust_price(0, seller.M)
                
            elif (self.price_adjust_method == 'step_random') or (self.price_adjust_method == 'proportional_random'):
                # Adjust buyer's price upwards
                pmax_buyer_  = self.adjust_price(pmax_buyer, np.random.uniform(0,self.alpha))
                # Adjust seller's price downwards
                pmin_seller_ = self.adjust_price(pmin_seller, -np.random.uniform(0,self.alpha))
                
            
            pmin_seller_  = max(pmin_seller_, self.minimum_price)
            
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
            
            ## EXCHANGE FUNCTION ##
            self._exchange(j, pmax_buyer, pmin_seller, buyer, seller)
            #######################
            
            # After exchange occurs, adjust prices for next round
            # Adjust buyer's price downwards
            #pmax_buyer_   = self.adjust_price(pmax_buyer, -self.beta) #(1-self.beta)*pmax_buyer
            
            # Adjust seller's price upwards
            #pmin_seller_  = self.adjust_price(pmin_seller, self.beta) #(1+self.beta)*pmin_seller
            
            # Adjust buyer's price upwards
            if (self.price_adjust_method == 'step') or (self.price_adjust_method == 'proportional'):
                pmax_buyer_  = self.adjust_price(pmax_buyer, -self.beta) #(1+self.alpha)*pmax_buyer
                # Adjust seller's price downwards
                pmin_seller_ = self.adjust_price(pmin_seller, self.beta) #(1-self.alpha)*pmin_seller
                
            elif (self.price_adjust_method == 'uniform'):
                pmax_buyer_ = self.adjust_price(0, buyer.M)
                pmin_seller_ = self.adjust_price(0, seller.M)
                
            elif (self.price_adjust_method == 'step_random') or (self.price_adjust_method == 'proportional_random'):
                pmax_buyer_  = self.adjust_price(pmax_buyer, -np.random.uniform(0,self.beta))
                pmin_seller_ = self.adjust_price(pmin_seller, np.random.uniform(0,self.beta))
                
            pmax_buyer_   = max(pmax_buyer_, self.minimum_price)
            
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
        if random.random() < self.epsilon_switch:
            pmax_buyer_ = pmin_seller
            if self.verbose:
                print('Switch Buyer Price to Seller Price {pmax_buyer_:.3f}')
                
        # Randomly reassign buyer price
        if random.random() < self.epsilon_rand:
            pmax_buyer_ = self.adjust_price(0, buyer.M, method='uniform')
            
        # Randomly switch seller price with buyer
        if random.random() < self.epsilon_switch:
            pmin_seller_ = pmax_buyer
            if self.verbose:
                print('Switch seller Price to Buyer Price {pmax_seller_:.3f}')
                
        # Random reassign seller price
        if random.random() < self.epsilon_rand:
            pmin_seller_ = self.adjust_price(0, seller.M, method='uniform')
        
            
        return pmax_buyer_, pmin_seller_
    
    def _exchange(self, j, pmax_buyer, pmin_seller, buyer, seller):
        
        # Select price
        p = np.random.uniform(pmax_buyer, pmin_seller)
        
        if self.verbose:
            print('     <<_exchange()>>')
            print('     -Exchanging-')
            print(f'     Commodity {j}')
            print()
            print(f'     pmax_buyer:     {pmax_buyer:.3f}')
            print(f'     pmin_seller:    {pmin_seller:.3f}')
            print(f'     Settled Price:  {p}')
            print()
            print('      Before Exchange')
            print(f'     Buyer {buyer}')
            print(f'     Buyer Money:    {buyer.M:.3f}')
            print(f'     Buyer D:        {buyer.D}')
            print(f'     Buyer Q:        {buyer.Q}')
            print(f'     Buyer Exchange: {buyer.D - buyer.Q}')
            print(f'                     {self.IntendedExchange[buyer]}')
            print()
            print(f'     Seller {seller}')
            print(f'     Seller Money:   {seller.M:.3f}')
            print(f'     Seller D:       {seller.D}')
            print(f'     Seller Q:       {seller.Q}')
            print(f'     Seller Exchange:{seller.D - seller.Q}')
            print(f'                     {self.IntendedExchange[seller]}')
            print()
            print(f'     -Swap Items {j}-')
            
            
        # Include the quantity and money exchanged for commodity j for this transaction
        _n = self.quantity_exchanged[j]
        _P = self.monetary_exchanged[j]
        _m = _P/_n if _n > 0 else 0# ERROR WHEN _n = 0!
        
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

        
        
        
        
        if self.verbose:
            print(self.printQ(self.IntendedExchange))
            print(f'Buyer Q: {buyer.Q}')
            print(f'Buyer M: {buyer.M}')
            print(f'Seller Q: {seller.Q}')
            print(f'Seller M: {seller.M}')
            
        buyer.M = buyer.M - p
        buyer.Q[j] = buyer.Q[j] + 1
  
        seller.M = seller.M + p
        seller.Q[j] = seller.Q[j] - 1
        
        if self.verbose:
            print(f'Buyer Q: {buyer.Q}')
            print(f'Buyer M: {buyer.M}')
            print(f'Seller Q: {seller.Q}')
            print(f'Seller M: {seller.M}')
            print(self.printQ(self.IntendedExchange))
                    
            print(self.IntendedExchange)
            
        # Update the IntendedExchange dictionary for the relevant agents
        self.IntendedExchange[buyer] = (buyer.D - buyer.Q)
        self.IntendedExchange[seller] = (seller.D - seller.Q)
        
        if self.verbose:
            print(self.IntendedExchange)
        
        if self.verbose:
            
           print('      After Exchange')
           print(f'     Buyer           {buyer}')
           print(f'     Buyer Money:    {buyer.M:.3f}')
           print(f'     Buyer D:        {buyer.D}')
           print(f'     Buyer Q:        {buyer.Q}')
           print(f'     Buyer Exchange: {buyer.D - buyer.Q}')
           print(f'                     {self.IntendedExchange[buyer]}')
           print()
           print(f'     Seller          {seller}')
           print(f'     Seller Money:   {seller.M:.3f}')
           print(f'     Seller D:       {seller.D}')
           print(f'     Seller Q:       {seller.Q}')
           print(f'     Seller Exchange:{seller.D - seller.Q}')
           print(f'                     {self.IntendedExchange[seller]}')
           print('                ------------                  ')
           print()
        
        # Update the agents bank attempts DONE IN OUTSIDE FUNCTION
        #self.AgentAttempts[buyer] = self.AgentAttempts[buyer] + 1
        #self.AgentAttempts[seller] = self.AgentAttempts[seller] + 1

    def attempt_exchange(self):
        
        # Pick a random uncleared commodity
        # Select random buyer and seller
        # Attempt offer price, and market exchange
        # Perform exchange if possible
        
        if self.verbose:
            print('<<attempt_exchange>>')
        
        if self.verbose:
            print('Q of Agents')
            self.printQ(self.IntendedExchange)
            print()
            print('M of Agents')
            self.printM(self.IntendedExchange)
            print()
            
        # Pick a random uncleared commodity
        j = random.choice(self.uncleared_commodities)

        # For commodity j create the set of buyers and sellers
        # Buyers have a positive intended exchange (want to increase their commodity j)
        # Sellers have a negative intended exchange (want to decrease their commodity j)
        BuyerSet = {n:E[j] for (n,E) in self.IntendedExchange.items() if E[j] > 0}
        SellerSet = {n:E[j] for (n,E) in self.IntendedExchange.items() if E[j] < 0}
        
        if self.verbose:
            print('Buyer Set')
            print(BuyerSet)
            print()
            print('Seller Set')
            print(SellerSet)
            print()
        
        if self.verbose:
            print('<<attempt_exchange()>>')
            print('--Attempt An Exchange--')
            print(f'Commodity {j}')
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
                print(f'Buyer D:   {buyer.D:}')
                print(f'Buyer Q:   {buyer.Q:}')
                print(f'Buyer E:   {buyer_E:.0f}')
                print(f'           {buyer.D - buyer.Q}')
                print(f'           {self.IntendedExchange[buyer]}')
                print()
                print(f'Seller:    {seller}')
                print(f'Seller D:  {seller.D}')
                print(f'Seller Q:  {seller.Q}')
                print(f'Seller E:  {seller_E:.0f}')
                print(f'           {seller.D - seller.Q}')
                print(f'           {self.IntendedExchange[seller]}')
                print()
                print('Set Prices and Try an Exchange... <<set_prices_exchange()>>')
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
                print('Return to <<attempt_exchange()>>')
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
                print(self.IntendedExchange)
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
            
            if self.verbose:
                print(self.IntendedExchange)
            #display(f'Commodities {self.uncleared_commodities}')
            #print(self.n_tries, self.max_tries)
            self.attempt_exchange()
            self.n_tries += 1
            if self.verbose:
                print(f'Total Exchange Attempts {self.n_tries}/{self.max_tries}')
                print()
            
        if self.verbose:
            print('            ------------                      ')
            
        #avg_price = self.monetary_exchanged/self.quantity_exchanged
        #std_price = self.s2
        
        #print()
        
        return self.monetary_exchanged, self.quantity_exchanged, self.s2, self.n_tries
 
        