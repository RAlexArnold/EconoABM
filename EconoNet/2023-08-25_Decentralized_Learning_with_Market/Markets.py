# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 19:13:42 2023

@author: Alex
"""

import numpy as np
import random

class SimpleMarket():
    
    def __init__(self, Nproducts, *, max_tries = 1000):
        
        # Initialize
        self.Nproducts = Nproducts
        
        self.n_tries = 0
        self.max_tries = max_tries
        
        self.IntendedExchange = {}
        
        self.uncleared_commodities = list(range(Nproducts))        
        self.quantity_exchanged = np.zeros(shape=Nproducts)# * np.nan #np.arange(Nproducts)
        self.monetary_exchanged = np.zeros(shape=Nproducts)# * np.nan #np.arange(Nproducts)
  
        self.price = None   
        
    def reset(self):
        
        self.n_tries = 0
        self.monetary_exchanged = np.zeros(shape=self.Nproducts)
        self.quantity_exchanged = np.zeros(shape=self.Nproducts)
  
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

        IntendedExchange = {}

        for Agent in ExchangeDict:

            E_ = Agent.D - Agent.Q
            #print(Agent, Agent.D, Agent.Q, E_)

            IntendedExchange[Agent] = E_

        return IntendedExchange
    
    def price_offer_rule_1(self, agent):
        return random.random()*agent.M
    
    def market_exchange_rule_1(self, dM_buyer, dM_seller):
        return random.uniform(dM_buyer, dM_seller)
    
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
        
        #print('---Market---')
        #print(j)
        #print('------------')
        #display(f'Buyers {BuyerSet}')
        #display(f'Sellers {SellerSet}')
        #print()

        # If BuyerSet or SellerSet are empty, then remove j from market
        # OR in this timestep there is only 1 market iteration for each agent
        # OR we define a 'speed' v of market, so that there are max ~v*dt market iterations
        # Regardless, if Buyer or Seller Set are empty, then j is removed from market
        if (len(BuyerSet) == 0) or (len(SellerSet) == 0):
            self.uncleared_commodities.remove(j)

        else:

            # Select random buyer
            buyer, buyer_E = random.choice(list(BuyerSet.items()))

            # Select random seller
            seller, seller_E = random.choice(list(SellerSet.items()))

            # The seller won't sell more than they are willing
            # And the buyer won't buy more than they are willing
            # So the (absolute) minimum of either will limit the quantity exchanged
            E_seller_to_buyer = min(-seller_E, buyer_E)

            # Price Offer Rules
            # Very simple uniform distribution (O1)
            # Future versions can include more complicated offer rule based off mean prices
            # Buyer offer price for commodity j
            dM_buyer = self.price_offer_rule_1(buyer) #random.random()*buyer.M

            # Seller offer price for commodity j
            dM_seller = self.price_offer_rule_1(seller) #random.random()*seller.M

            # Market Exchange Rules
            # Very simple uniform distribution (E1)
            # Future versions could include more complicated exchange rules based off each trying to maximize/minimize the price
            dM_buyer_to_seller = self.market_exchange_rule_1(dM_buyer, dM_seller) #random.uniform(dM_buyer, dM_seller)
            
            #display(f'Buyer {buyer} E* {buyer_E} M {buyer.M}')
            #display(f'Seller {seller} E* {seller_E}')
            #print()
            #display(f'dM {dM_buyer_to_seller}')
            #display(f'E {E_seller_to_buyer}')
            #print()

            # Exchange is successful if the buyer has the sufficient funds
            if dM_buyer_to_seller <= buyer.M:
                
                #print('Accomplish Sale')
                #print()
                
                # Include the quantity and money exchanged for commodity j for this transaction
                # At end of market run this will be used to find an average price
                self.quantity_exchanged[j] = self.quantity_exchanged[j] + abs(E_seller_to_buyer)
                self.monetary_exchanged[j] = self.monetary_exchanged[j] + abs(dM_buyer_to_seller)


                buyer.M = buyer.M - dM_buyer_to_seller
                seller.M = seller.M + dM_buyer_to_seller

                buyer.Q[j] = buyer.Q[j] + E_seller_to_buyer
                seller.Q[j] = seller.Q[j] - E_seller_to_buyer

                # Update the IntendedExchange dictionary for relevant agents
                self.IntendedExchange[buyer] = buyer.D - buyer.Q
                self.IntendedExchange[seller] = seller.D - seller.Q      
        
    def run_exchange(self, ExchangeDict):
        
        self.quantity_exchanged[:] = 0
        self.monetary_exchanged[:] = 0
        
        self.n_tries = 0
        
        # Create the dictionary of agents and their intended exchanges
        self.IntendedExchange = self.set_intended_exchange(ExchangeDict)
        
        # Create a list of commodities still on the market (uncleared)
        self.uncleared_commodities = list(range(self.Nproducts))
        
        # Loop while buyer and seller non empty (len[uncleared_commodites] > 0) or if n_tries > max_tries
        while (len(self.uncleared_commodities) > 0) and (self.n_tries < self.max_tries):
            
            #print(len(self.uncleared_commodities), self.n_tries, self.max_tries)
            
            #display(f'Commodities {self.uncleared_commodities}')
            self.attempt_exchange()
            self.n_tries += 1
            
        #if self.quantity_exchanged
        price = self.monetary_exchanged/self.quantity_exchanged
        #print(self.monetary_exchanged, self.quantity_exchanged)
        
        self.price = price
        
        #return price, self.monetary_exchanged, self.quantity_exchanged, self.n_tries
            
            
        
        
    
    
        