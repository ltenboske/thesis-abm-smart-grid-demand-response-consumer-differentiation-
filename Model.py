#!/usr/bin/env python
# coding: utf-8

# # Import packages

# In[1]:

# import basic packages
import numpy as np
import pandas as pd
# import itertools
import random
import queue

import networkx as nx
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# import required mesa packages
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid

# input data 
df_demand = pd.read_csv('df_demand.csv')
df_production = pd.read_csv('df_production.csv')
yhat_pred = pd.read_csv('yhat_pred.csv')
yhat_pred = list(yhat_pred.iloc[:,0])

green_list_year = np.array(df_production['renewable_generation_kWh'])  # green production in kWh for each t
fossil_list_year = np.array(df_production['fossil_generation_kWh']) # fossil production in kWh for each t

year_dem = df_demand.iloc[:35019,:] 
year_prod = df_production.iloc[:35011,:]
green_prediction = yhat_pred[:35020]  

demand_list = np.array(year_dem['E1A_kWh'])
shift_list = np.array(year_dem['shiftable_%']) / 1.5  # heating  #about 40%
semishift_list = (np.array(year_dem['semi_shiftable_%']) + np.array(year_dem['shiftable_%'])) / 1.5 # heating and semi-shiftable demand # about 50% 
green_list = np.array(year_prod['renewable_generation_kWh'])  # green production in kWh for each t
fossil_list = np.array(year_prod['fossil_generation_kWh']) # green production in kWh for each t

########################################################################################################

def barabasi_albert_adapted(n, m, types, types_division, seed=None): 
    '''BA is preferential attachment model based on a node's degree '''
    '''Extended with preferencial attachment to the same consumer type'''

    if m < 1 or m >= n:
            raise nx.NetworkXError(
                f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
            )

    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.empty_graph(m)
    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    s = range(0,source)
    # Create the first node in the network
    for i in s:
        typ = types[types_division[i]]
        G.add_node(i, agent_type = typ)
        G.add_node(i, color = 'grey')
    # Create the other nodes
    while source < n:
        typ = types[types_division[source]]
        G.add_node(source, agent_type = typ)
        # Determine node color based on consumer type
        for node, attr in G.nodes(data=True):
            if attr['agent_type'] == 'green':
                G.add_node(source, color = 'green',label ='green')
            elif attr['agent_type'] == 'cost-conscious':
                G.add_node(source, color = 'blue', label = 'cost-conscious')
            elif attr['agent_type'] == 'convenience-oriented':
                G.add_node(source, color = 'yellow',label='convenience-oriented')
            else:
                G.add_node(source, color = 'red', label='indifferent')
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list. 
        repeated_nodes.extend([source] * m)
        # check own type
        own_type = nx.get_node_attributes(G, "agent_type")
        ot = own_type[source]
       # list of all nodes with the same type
        nodes_same_type = [x for x,y in G.nodes(data=True) if y['agent_type']==ot]
        nodes_same_type.remove(source)  #remove yourself from the list
        # add nodes of same type to the choice options list
        choice_list = repeated_nodes + nodes_same_type + nodes_same_type  
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = random.sample(choice_list, m)
        source += 1
     
    # Save degree as node attribute
    degrees = {node:val for (node, val) in G.degree()}
    nx.set_node_attributes(G, degrees, 'degree')
        
    return G

####################### model metrics #################################

"Model outcomes"    
def costs(model):   #calculates total costs each time step
    agents_cost = [agent.cost for agent in model.schedule.agents]
    total_cost = sum(agents_cost)
    return total_cost

def demand_total(model):    #calculates total demand (shifting included) each time step
    agents_demand = [agent.demand for agent in model.schedule.agents]
    total_demand = sum(agents_demand)
    return total_demand


#################### agent setup ###############################

"define the consumer" 
class Consumer(Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, unique_id, model, consumer_type, init_package):
        super().__init__(unique_id, model)
        # initialize consumer type
        self.consumer_type = consumer_type
        self.init_package = init_package
        
        # initialize personal values
        self.value_price = self.set_value_price(self.consumer_type)
        self.value_environment = self.set_value_environment(self.consumer_type)
        self.value_comfort = self.set_value_comfort(self.consumer_type)
        self.value_safety = self.set_value_safety(self.consumer_type)
        self.value_social_norm = self.set_value_social_norm(self.consumer_type)
        self.participation_score = self.value_price + self.value_environment - self.value_comfort - self.value_safety
        self.social_pressure = 0
        self.influential_neighbor = None
        self.threshold_shift = model.threshold_shift
        
        # intitialze values for demand shifting
        self.shift_option = self.set_RTP_shift_option(self.consumer_type)
        self.shift_perc = []
        self.answer = ' '
        self.demand = demand_list[0]
        self.demand_history = []
        self.shift_history_real = []
        self.shifted_demand = []
        self.shift_residual = 0
        self.score_history = []  
        self.cost = model.price * self.demand
        
        # initialize policies
        self.ec = 0
        self.sn = 0 
        self.informed_environment = 0
        self.informed_environment_time = 0 
        self.informed_smart_meter = 0
        self.informed_smart_meter_time = 0
        self.env_score = queue.Queue(self.model.switch_period)
        self.env_score_prev_period = queue.Queue(self.model.switch_period)


###################### agent functions for initializing agents ################### 

    def set_value_price(self,consumer_type): 
        p = {"green": round(0.35 + (random.randrange(-10, 10,1) / 100),2),
            "cost-conscious": round(0.9 + (random.randrange(-10, 10,1) / 100),2),
            "convenience-oriented": round(0.35 + (random.randrange(-10, 10,1) / 100),2),
            "indifferent": round(0.1 + (random.randrange(-10, 10,1) / 100),2)}
        ps = p[consumer_type] 
        return ps
    
    def set_value_environment(self,consumer_type): 
        e = {"green": round(0.9 + (random.randrange(-10, 10,1) / 100),2),          
            "cost-conscious": round(0.35 + (random.randrange(-10, 10,1) / 100),2),
            "convenience-oriented": round(0.25 + (random.randrange(-10, 10,1) / 100),2),
            "indifferent": 0.0 + abs(round((random.randrange(-10, 10,1) / 100),2))}
        es = e[consumer_type] 
        return es
    
    def set_value_comfort(self,consumer_type): 
        c = {"green": round(0.1 + (random.randrange(-10, 10,1) / 100),2), 
            "cost-conscious": round(0.3 + (random.randrange(-10, 10,1) / 100),2),
            "convenience-oriented": round(0.8 + (random.randrange(-10, 10,1) / 100),2),
            "indifferent": round(0.9 + (random.randrange(-10, 10,1) / 100),2)}
        cs = c[consumer_type] 
        return cs

    def set_value_safety(self,consumer_type): 
        s = {"green": round(0.1 + (random.randrange(-10, 10,1) / 100),2),
            "cost-conscious": round(0.7 + (random.randrange(-10, 10,1) / 100),2),
            "convenience-oriented": round(0.5 + (random.randrange(-10, 10,1) / 100),2),
            "indifferent": round(0.2 + (random.randrange(-10, 10,1) / 100),2)}
        ss = s[consumer_type] 
        return ss
    
    def set_value_social_norm(self,consumer_type): 
        sn = {"green": round(0.65 + (random.randrange(-10, 10,1) / 100),2),
            "cost-conscious": round(0.65 + (random.randrange(-10, 10,1) / 100),2),
            "convenience-oriented": round(0.35 + (random.randrange(-10, 10,1) / 100),2),
            "indifferent": round(0.40 + (random.randrange(-10, 10,1) / 100),2)}
        sns = sn[consumer_type] 
        return sns 
    

    def set_RTP_shift_option(self,consumer_type):
        '''There are three packages: no shift, flex shift and semi-flex + flex shift'''
        packages = {1: "no shifting", 2: "shiftable", 3: "semi- and shiftable"} 
        if self.init_package == 'probabilistic':
            p = {"green": packages[int(np.random.choice([1,2,3], 1, p=[0.05, 0.45, 0.5]))],
                "cost-conscious": packages[int(np.random.choice([1,2,3], 1, p=[0.05, 0.60, 0.35]))],
                "convenience-oriented": packages[int(np.random.choice([1,2,3], 1, p=[0.60, 0.40, 0.0]))],
                "indifferent": packages[int(np.random.choice([1,2,3], 1, p=[0.9, 0.1, 0.0]))]}
            pack = p[consumer_type]
        elif self.init_package == 'default_shift': # shifting heating and airco  as default
            if self.participation_score > self.model.threshold_semishift:
                pack = packages[3]
            else:
                pack = packages[2]
        elif self.init_package == 'default_noshift': # no shifting and no switching (base case A)
            pack = packages [1]
        elif self.init_package == 'value_based':
            if self.participation_score >= self.model.threshold_semishift:
                pack = packages[3]
            elif self.participation_score >= self.model.threshold_shift:
                pack = packages[2]
            else:
                pack = packages[1]
        else: #init_package == 'type-based'    
            p = {"green": packages[3],
                "cost-conscious": packages[2],
                "convenience-oriented": packages[1],
                "indifferent": packages[1]}
            pack = p[consumer_type]
        return pack
    
 ##########################  agent functions at each time step ##########################
    
    def set_demand_history(self):   
        '''store original demand data in agent'''
        self.demand = demand_list[self.model.count] 
        self.demand_history.append(self.demand)  
        return self.demand_history, self.demand
        
    
    def calc_cost(self):
        '''calculate the cost for each agent each time step depending on shift package'''
        if self.shift_option == 'no shifting':
            self.cost = self.model.price * self.demand
            # store costs for non-shifter as model variable for cost comparison
            self.model.no_shift_cost = self.cost   
            self.model.no_shift_cost_history.append(self.model.no_shift_cost)
        elif self.shift_option == 'shiftable':
            self.cost = self.model.price * self.demand - self.model.reduction / self.model.switch_period
            # store costs for shifter as model variable for cost comparison
            self.model.shiftable_cost = self.cost
            self.model.shiftable_cost_history.append(self.model.shiftable_cost)
        else:   # reduction for a period for one time step
            self.cost = self.model.price * self.demand - self.model.reduction / self.model.switch_period
            # store costs for full shifter as model variable for cost comparison
            self.model.semishiftable_cost = self.cost
            self.model.semishiftable_cost_history.append(self.model.semishiftable_cost)
    
        return self.cost, self.model.no_shift_cost, self.model.shiftable_cost, self.model.semishiftable_cost, self.model.semishiftable_cost_history, self.model.no_shift_cost_history, self.model.shiftable_cost_history


    def social_network_neighbor_influence(self):
        '''initial network analysis to define social pressure'''
        '''find degree of neighbors and find out most influential neighbor'''
        if self.model.count == 1: 
            i = self.unique_id
            dic = {}
            # Who are the neighbors of agent i and what is their degree
            for neighbor in self.model.G.neighbors(i):  
                dic[neighbor] = self.model.G.nodes[neighbor]["degree"]
            # highest degree of neighbors 
            max_infl_nb = max(dic.values())
            # influence based on degree of other and own degree
            infl_nb = max(dic.values()) - self.model.G.nodes[i]["degree"]   

            # determine social pressure
            if self.model.G.nodes[i]["degree"] > infl_nb: 
                self.social_pressure = 0
            else: # 
                self.social_pressure = max(1,(infl_nb/2)) * self.value_social_norm 
    
                # determine most influential neighbor in case of multiple neighbors with max degree
                neighbors_with_max = [i for i in dic.keys() if max_infl_nb == dic[i]]
                if len(neighbors_with_max) > 1:
                    # neighbor with most comparable participation score is chosen
                    diff_PS = {n:abs(self.model.G.nodes[i]['participation_score'] - self.model.G.nodes[n]['participation_score']) for n in neighbors_with_max}
                    min_diff_PS = min(diff_PS.values())
                    self.influential_neighbor = list(diff_PS.keys())[list(diff_PS.values()).index(min_diff_PS)] 
                else: 
                    self.influential_neighbor = list(dic.keys())[list(dic.values()).index(max_infl_nb)] 

        return self.social_pressure, self.influential_neighbor
    
    
    def set_env_score(self):
        '''create environment score'''
        if self.model.green / self.model.num_agents > self.demand: # green supply is sufficient
            e = 1
        elif self.model.production_total / self.model.num_agents >= self.demand: # total supply is sufficient
            e = 0.5
        else: # energy shortage
            e = -1
    
        '''update score list'''
        # always add new score to queue    
        self.env_score.put(e)
        # check if it is a switching period again and empty queue
        if self.env_score.full() == True:
            self.env_score_prev_period = self.env_score # store full list
            self.env_score = queue.Queue(self.model.switch_period) # make a new empty queue
        
        return self.env_score, self.env_score_prev_period
    
    
    def set_participation_score(self): 
        '''determine the new participation score at the end of each switch period'''
        # calculated each switch period
        rounds = int((len(green_list)-1) / self.model.switch_period)
        values = list(range(1,rounds+1))
        switch_times = [(x * self.model.switch_period) for x in values]   
        
        if self.model.count in switch_times:
            # personal values
            PS = self.value_price 
            EC = self.value_environment 
            VC = self.value_comfort
            PR = self.value_safety
            SN = self.value_social_norm 

            '''Price''' 
            # check if the cost comparison intervention is in place
            if self.model.cost_comparison == 1:
                PS = (self.model.cost_difference / 100) * self.value_price + self.value_price 
            else:
                PS = self.value_price 
            
            '''Environment'''
            # Check if the environmental comparison intervention is in place
            if self.model.env_score_setting == 1:
                env_score_period = sum(list(self.env_score_prev_period.queue)) # the queue of the period length
                # compare score to others 
                diff_from_mean = env_score_period - self.model.mean_env_score  
                
                if diff_from_mean > 0: # you are doing good already
                    rel_env_score = 0 
                else: # you are doing less good   
                    rel_env_score = abs(diff_from_mean) / self.model.environment_compare_weight 
                # final environment score
                if self.model.count <= self.model.switch_period or diff_from_mean != 0: # it is the first switch period or there is a difference
                    EC = rel_env_score * self.value_environment + self.value_environment
                    self.ec = EC
                else:
                    EC = self.ec
            else:  # environmental comparison is not in place
                EC = self.value_environment
           
                       # environmental information campaign
            if self.informed_environment == 1:
                EC = EC * self.model.effect_info_campaign_environment  # increase of EC caused by campaign effect
            
            '''Social pressure'''
            if self.social_pressure != 0:
                # determine shifting contract of self and influential neighbor
                package_code = {'no shifting':0, 'shiftable':1, "semi- and shiftable":2}
                p_own = package_code[self.shift_option]  # own package
                shift_options_agents = [agent.shift_option for agent in self.model.schedule.agents]
                p_other = package_code[shift_options_agents[self.influential_neighbor]]
                # compare packages to determine if social pressure is negative or positive
                if p_other > p_own:
                    SN = self.social_pressure * self.model.social_pressure_effect 
                elif p_other < p_own: 
                    SN = - self.social_pressure * self.model.social_pressure_effect  
                else: # same contract
                    SN = self.sn 
            else: 
                 SN = 0
            # prevent switching back if 
            self.sn = SN
        
            '''Comfort and safety'''
            if self.informed_smart_meter == 1:    
                # decrease by campaign effect
                VC = self.value_comfort * self.model.effect_info_campaign_smart_meter
                PR = self.value_safety * self.model.effect_info_campaign_smart_meter 
                
            '''calculate final  participation score'''
            score = PS + EC - VC - PR + SN  #SN can be positive and negative
            self.score_history.append(score)
            self.participation_score = score
            
        return self.participation_score, self.score_history, self.ec, self.sn
    
    
    def package_shifting(self):
        '''Consumers can switch their shifting contract option each time period'''
        '''They switch if the participation that reaches one of the thresholds'''
        # switching is possible at the end of each switch period
        rounds = int((len(green_list)-1) / self.model.switch_period)
        values = list(range(1,rounds+1))
        switch_times = [x * self.model.switch_period for x in values]       
  
        # select the shifting contract by looking at the thresholds
        if self.model.count in switch_times and self.init_package != 'default_noshift':
            if self.participation_score < self.model.threshold_shift:
                self.shift_option = 'no shifting'
            elif self.model.threshold_shift < self.participation_score < self.model.threshold_semishift:
                self.shift_option = 'shiftable'
            else:
                self.shift_option = 'semi- and shiftable'
        else:  # keep old one if it is not a switching moment
            self.shift_option = self.shift_option
        
        return self.shift_option
    
    
    def demand_shifting_yes_or_no(self):
        '''The energy producer determines if demand shifting takes place''' 
        '''Only possible 'yes' if the consumer has chosen a shifting package that allows for shifting'''
        if self.shift_option == 'no shifting':
            answer = 'no'
        elif self.model.mismatch_prediction[1] < 0:   #Shift based on the mismatch prediction
            answer = 'yes'
        else:
            answer = 'no'
            
        self.answer = answer
        self.shift_history_real.append(self.answer)
    
        return self.answer, self.shift_history_real

        
    def shifting_the_demand(self):
        '''Shift the demand if the answer is yes'''
        '''Determine demand and shifted demand for each agent'''
        '''Looks at the the current time step and previous 1,5 hour (max shift time)'''
        # determine the percentages that can be shifted at this time
        if self.shift_option == 'shiftable':
            self.shift_perc = list(shift_list[self.model.count-7:self.model.count])
            if self.model.count < 7: 
                self.shift_perc = list(shift_list[:7])                
        elif self.shift_option == 'semi- and shiftable':
            self.shift_perc = list(semishift_list[self.model.count-7:self.model.count]) 
            if self.model.count < 7: 
                self.shift_perc = list(semishift_list[:7])      
        else: # self.shift_option == 'no shifting':
            self.shift_perc = [0,0,0,0,0,0,0]
            
        # answer 'yes' or 'no' for shifting 
        history_now = self.shift_history_real[-7:] 
        if len(history_now) < 7:
            history_now = ['no','no','no','no','no','no','no'] # t-6 until t
        history = history_now[:6]  # t-6 until t-1
        # original demand data 
        previous_demand = self.demand_history[-7:]    
        if len(previous_demand) < 7:
            previous_demand = list(demand_list[:self.model.count+1])
        # shifted demand from previous 6 time steps   
        previous_shift = self.shifted_demand[-7:]     
        if len(previous_shift) < 7:
            previous_shift = [0,0,0,0,0,0,0]
        previous_shift = previous_shift[-6:]
     
        # calculate energy amount that a shifter should shift to solve shortage comletely
        # it is the maximum amount that an agent should shift
        if self.model.mismatch_total < 0 and self.shift_option == 'shiftable':
            shiftshift = self.model.shifters_share * self.model.mismatch_total
        elif self.model.mismatch_total < 0 and self.shift_option == 'semi- and shiftable':
            shiftshift = self.model.semishifters_share * self.model.mismatch_total
        else: 
            shiftshift = 0
                
        '''loop through 7 time steps'''
        '''important assumption: demand can only be shifted 6 time steps = 1.5 hour to limit inconveniences'''
        # No demand is shifted in the first 7 time steps and if an agent has a no shifting contract
        if self.model.count < 7 or self.shift_option == 'no shifting':
            demand_now = demand_list[self.model.count]
            all_demand_now = demand_now
            self.shifted_demand.append(0)
        
        # for shifters and full shifters
        # find index of last occurence of 'no'
        elif 'no' in history:  # current shortage period is shorter than 6 time step (1,5 hour)
            idx = next(i for i in reversed(range(len(history))) if history[i] == 'no')
            # determine demand now
            if history_now[-1] == 'no': # no shifting in the current time step, so there is no shortage
                # a1 = shifted demand from previous time steps
                a1 = np.array(previous_shift[(idx+1):])
                # current demand + previous shifted demand
                all_demand_now = sum(a1) + previous_demand[-1] 
                # calculate available supply for this agent, considering that some others do not shift
                production_one_agent = self.model.production_total / self.model.num_agents  
                # cannot use more than production_one_agent
                demand_now = min(all_demand_now,production_one_agent) 
                # part of all demand now that could not be used now goes to shift residual 
                self.shift_residual = all_demand_now - demand_now + self.shift_residual
                # no newly shifted demand at this time step
                self.shifted_demand.append(0) 
            else: # there is shortage, demand should be shifted in this time step
                # non-shiftable share of current demand 
                all_demand_now = previous_demand[-1] * (1 - self.shift_perc[-1])  
                # calculate available supply for this agent, considering that some others do not shift
                production_one_agent = demand_list[self.model.count] + shiftshift
                # minimally use the available supply and prevent unnecessary shifting
                if production_one_agent >= all_demand_now:
                    demand_now = production_one_agent
                    shifted_demand = previous_demand[-1] - production_one_agent
                else: # shift maximum possible if production one agent < all demand now to limit shortage
                    demand_now = all_demand_now 
                    shifted_demand = previous_demand[-1] * self.shift_perc[-1]  
                self.shifted_demand.append(shifted_demand) # shifted amount from this time step
        
        else: # current shortage period is longer than 6 time step (1,5 hour), so shifted demand from t-6 should be used now
            if history_now[-1] == 'yes': # there is shortage, demand should be shifted in this time step
                # shifted demand from 6 time steps ago + non-shiftable share of current demand
                all_demand_now = previous_shift[0] + previous_demand[-1] * (1 - self.shift_perc[-1])
                # calculate available supply for this agent, considering that some others do not shift
                production_one_agent = demand_list[self.model.count] + shiftshift  
                # minimally use the available supply and prevent unnecessary shifting
                if production_one_agent > all_demand_now:
                    demand_now = production_one_agent
                    shifted_demand = previous_demand[-1] - production_one_agent + previous_shift[0]
                else: # shift maximum possible if production one agent < all demand now to limit shortage
                    demand_now = all_demand_now
                    shifted_demand = previous_demand[-1] * self.shift_perc[-1]
                self.shifted_demand.append(shifted_demand) # shifted amount from this time step
            else: # end of the long shortage period, no new demand should be shifted anymore
                 # a1 = shifted demand from previous time steps
                a1 = np.array(previous_shift[:6])
                # current demand + previous shifted demand
                all_demand_now = sum(a1) + previous_demand[-1] 
                 # calculate available supply for this agent, considering that some others do not shift
                production_one_agent = self.model.production_total / self.model.num_agents
                # cannot use more than production_one_agent
                demand_now = min(all_demand_now,production_one_agent) 
                # part of all demand now that could not be used now, goes to shift residual
                self.shift_residual = all_demand_now - demand_now + self.shift_residual
                # no newly shifted demand at this time step
                self.shifted_demand.append(0) 
                
        # store demand that is used now after shifting as agent variable
        self.demand = demand_now
        
        '''Check if the shift residual can be used in the time steps after a shortage period'''
        '''Shift residual should be used within 6 time steps, otherwise from storage'''
        # calculate extra available supply that is available for a shifter 
        if self.model.real_mismatch_total > 0 and self.shift_option == 'shiftable': 
            extra_demand = self.model.shifters_share * self.model.real_mismatch_total
        elif self.model.real_mismatch_total > 0 and self.shift_option == 'semi- and shiftable':
            extra_demand = self.model.semishifters_share * self.model.real_mismatch_total
        else: 
            extra_demand = 0
        
        # check if current demand is smaller than available supply 
        if self.shift_residual > 0 and self.demand < (self.model.production_total / self.model.num_agents + extra_demand):
            # how much of the shift residual could potentially be used now
            potential = (self.model.production_total / self.model.num_agents + extra_demand) - self.demand 
            # use the potential amount or the remaining shift residual
            self.demand = self.demand + min(potential, self.shift_residual)  
            # decrease shift residual
            self.shift_residual = self.shift_residual - min(potential, self.shift_residual) 
        # use all shift residual after 6 time steps
        if self.shift_residual > 0 and all(x=='no' for x in history):
            self.demand = self.demand + self.shift_residual
            self.shift_residual = 0 
                                                     
        # make sure that no shifters never shift
        if self.model.count < 7 or self.shift_option == 'no shifting':
            self.demand = demand_list[self.model.count]

        return self.demand, self.shifted_demand, self.shift_residual
    
    #################### agent step ###################

    def step(self):
        self.social_network_neighbor_influence()
        self.set_demand_history()
        self.set_participation_score()
        self.package_shifting()
        self.demand_shifting_yes_or_no()
        self.shifting_the_demand()
        self.calc_cost()        
        self.set_env_score()


####################################### initialize the model #############################################
        
'''define the model''' 
class GridModel(Model):
    """A model with some number of agents."""
    def __init__(self, green_list_year, fossil_list_year, green_list, fossil_list, semishift_list, shift_list, demand_list, green_prediction, price_flat, alpha, beta, 
                 gamma, RTP_policy, switch_period, threshold_shift, threshold_semishift, social_pressure_effect, seed, N, population_options, population_division, N_green, N_cost, N_convenience,
                 N_indifferent, supply_factor, cost_comparison, reduction, env_score_setting, long_term_effect_info, n_info_campaign_environment, effect_info_campaign_environment, n_info_campaign_smart_meter,
                 effect_info_campaign_smart_meter, init_package):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.running = True
        
        # init model parameters regarding demand, supply, price and storage
        self.supply_factor = supply_factor
        self.switch_period = switch_period
        self.init_package = init_package
        self.count = 0
        self.green = green_list[0] * supply_factor
        self.fossil = fossil_list[0]
        self.shifters_share = 0
        self.semishifters_share = 0
        self.share_noshift = 0
        self.share_shift = 0
        self.share_semiandshift = 0
        self.threshold_semishift = threshold_semishift
        self.threshold_shift = threshold_shift
        self.RTP_policy = RTP_policy
        self.price_flat = price_flat
        self.price = price_flat
        self.price_history = []
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.production_total = green_list[0] * supply_factor + fossil_list[0]
        self.generation_mean = (sum(green_list_year*self.supply_factor) + sum(fossil_list_year)) / len(green_list_year)
        self.green_prediction = np.array(green_prediction) * self.supply_factor
        self.prediction_total_production = np.array(self.green_prediction[:10]) * self.num_agents + fossil_list[:10]
        self.mismatch_total = 0  
        self.real_mismatch_total = 0 
        demand_list = demand_list
        self.mismatch_prediction = np.array(self.prediction_total_production) - np.array(demand_list[self.count:self.count+10]) * self.num_agents
        self.total_demand = 0
        self.storage_capacity = 100 #kwh
        self.storage_soc = 0
        self.energy_shortage = 0
        
        # activate policies
        self.no_shift_cost = 0
        self.no_shift_cost_history = []
        self.shiftable_cost = 0 
        self.shiftable_cost_history = []
        self.semishiftable_cost = 0
        self.semishiftable_cost_history = []
        self.cost_difference = 0
        self.reduction = reduction
        self.n_info_campaign_environment = n_info_campaign_environment
        self.informed_env = 0
        self.n_info_campaign_smart_meter = n_info_campaign_smart_meter
        self.informed_sm = 0
        self.mean_env_score = 0
        self.max_env_score = 0
        self.environment_compare_weight = 50
        self.env_score_setting = env_score_setting
        self.cost_comparison = cost_comparison
        self.social_pressure_effect = social_pressure_effect
        self.effect_info_campaign_environment = effect_info_campaign_environment
        self.effect_info_campaign_smart_meter = effect_info_campaign_smart_meter
        self.long_term_effect_info = long_term_effect_info
        
        # create all potential options of  division of population
        def sums(length, total_sum):
            if length == 1:
                yield (total_sum,)
            else:
                for value in range(total_sum + 1):
                    for permutation in sums(length - 1, total_sum - value):
                        yield (value,) + permutation

        if population_options == 'all':
            L = list(sums(4,N))
        else: # interesting population settings for a population of 20
            L = [[20,0,0,0],[0,20,0,0],[0,0,20,0],[0,0,0,20],[10,10,0,0],[10,0,10,0],[10,0,10,0],[0,10,10,0],[0,0,10,10],[10,0,0,10],[0,10,0,10],
                 [5,5,0,10],[5,5,5,5],[5,5,10,0],[5,0,5,10],[0,5,5,10],[10,5,0,5],[5,10,0,5],[0,10,5,5],[10,0,5,5],[10,5,0,5],[10,5,5,0],
                 [0,5,0,15],[15,0,0,5],[0,15,0,5],[5,0,15,0],[0,5,15,0],[0,15,0,5],[5,0,0,15],[5,15,0,0],[15,0,5,0],[15,5,0,0],
                 [18,0,0,2],[0,18,0,2],[2,0,0,18],[0,2,0,18],[1,1,0,18],[18,1,1,0],[0,18,2,0],[2,0,18,0],[9,9,1,1],[1,1,9,9]]

        
        # create random shuffled list of types in population
        chosen_division = L[population_division]
        types = {1: "green", 2: "cost-conscious", 3: "convenience-oriented", 4: "indifferent"}
        types_division = [1]*chosen_division[N_green] + [2]*chosen_division[N_cost] + [3]*chosen_division[N_convenience] + [4]*chosen_division[N_indifferent]
        random.shuffle(types_division)

        #Create agents
        self.G = barabasi_albert_adapted(self.num_agents, 1, types, types_division)
        
        for i, node in enumerate(self.G.nodes()): 
            t = types_division[i]
            typ = types[t] 
            init_package = self.init_package
            
            a = Consumer(i, self, typ, init_package)
            # place agent in the social network
            self.G.add_node(i, participation_score = a.participation_score)
            self.schedule.add(a)
            self.grid = NetworkGrid(self.G)
            self.grid.place_agent(a,node) 

        # collect data about agent and model variables
        self.datacollector = DataCollector(
            model_reporters={"Total Demand": demand_total,"Total Energy Costs": costs,"energy shortage":"energy_shortage",
                            "Total Production":"production_total","storage SoC":'storage_soc',"Non-shifters share":"share_noshift",
                            "Shifters share":"share_shift","Full shifters share":"share_semiandshift"},
            agent_reporters={"Type":"consumer_type","Participation score":"participation_score"})

    
########################## Function in model step ######################

    def count_steps(self):
        '''counts the time step in the model'''
        self.count += 1
        return self.count
        
    def green_supply(self):   #calculate green supply
        self.green = green_list[self.count] * self.num_agents * self.supply_factor
        return self.green
    
    def fossil_supply(self):
        self.fossil = fossil_list[self.count] * self.num_agents
        return self.fossil
    
    def shifting_population_division(self):
        '''calculate the shares of shifting contracts in the population'''
        self.share_shift =  len([agent for agent in self.schedule.agents if agent.shift_option == 'shiftable']) / self.num_agents
        self.share_semiandshift =  len([agent for agent in self.schedule.agents if agent.shift_option == 'semi- and shiftable']) / self.num_agents
        self.share_noshift =  len([agent for agent in self.schedule.agents if agent.shift_option == 'no shifting']) / self.num_agents

        #calculate how much of total mismatch 1 agent should shift, considering that some agents do not shift demand 
        if self.share_shift + self.share_semiandshift > 0 and self.share_noshift > 0: # there are agents with each type of contract
            if self.share_shift > 0:
                self.shifters_share = (self.share_shift + (self.share_shift / (self.share_shift + self.share_semiandshift)) * self.share_noshift) / len([agent for agent in self.schedule.agents if agent.shift_option == 'shiftable'])
            if self.share_semiandshift > 0:
                self.semishifters_share = (self.share_semiandshift + (self.share_semiandshift / (self.share_shift + self.share_semiandshift)) * self.share_noshift) / len([agent for agent in self.schedule.agents if agent.shift_option == 'semi- and shiftable'])
        elif self.share_shift + self.share_semiandshift > 0 and self.share_noshift == 0:  # there are only shifters and full shifters
            if self.share_shift > 0:
                self.shifters_share = self.share_shift / len([agent for agent in self.schedule.agents if agent.shift_option == 'shiftable'])   
            if self.share_semiandshift > 0:
                self.semishifters_share = self.share_semiandshift / len([agent for agent in self.schedule.agents if agent.shift_option == 'semi- and shiftable'])   
        else: # there are only no shifters
            self.shifters_share = 0
            self.semishifters_share = 0
        
        return self.shifters_share, self.semishifters_share
    
    def total_production_calc(self):
        self.production_total = self.green + self.fossil
        return self.production_total
    
    def total_production_predict(self): 
        '''prediction of production next 10 time steps'''
        prediction_total_production = np.array(self.green_prediction[self.count:self.count+10]) * self.num_agents * self.supply_factor
        prediction_total_production = prediction_total_production + self.fossil
        self.prediction_total_production = prediction_total_production
        return self.prediction_total_production
    
    def total_mismatch_predict(self):  
        '''prediction of shortage next 10 time steps'''
        prediction_demand = np.array(demand_list[self.count:self.count+10]) * self.num_agents
        self.mismatch_prediction = np.array(self.prediction_total_production) - np.array(prediction_demand)
        return self.mismatch_prediction 
    
    def demand_total(self): 
        '''calculates total demand each time step'''
        agents_demand = [agent.demand for agent in self.schedule.agents]
        total_demand = sum(agents_demand)
        self.total_demand = total_demand
        return self.total_demand
    
    
    def information_campaign_environment(self):
        # select the consumers that are reached by the campaign
        nonshifters = [agent for agent in self.schedule.agents if agent.shift_option == 'no shifting'] 
        nonshifter_sample = self.random.sample(nonshifters, min(len(nonshifters), self.n_info_campaign_environment))
        # campaign effect disappears after a while
        for agent in self.schedule.agents:
            if self.count - agent.informed_environment_time > self.long_term_effect_info and agent.informed_environment == 1:
                agent.informed_environment = 0    
        # set informed state to 1 for selected consumers
        for agent in nonshifter_sample:
            agent.informed_environment = 1 
            agent.informed_environment_time = self.count
        informed_env = [agent for agent in self.schedule.agents if agent.informed_environment == 1]
        self.informed_env = len(informed_env)
            
        
    def information_campaign_smart_meter(self):
        # select the consumers that are reached by the campaign
        nonshifters = [agent for agent in self.schedule.agents if agent.shift_option == 'no shifting'] 
        nonshifter_sample = self.random.sample(nonshifters, min(len(nonshifters), self.n_info_campaign_smart_meter))
        # campaign effect disappears after a while
        for agent in self.schedule.agents:
            if self.count - agent.informed_smart_meter_time > self.long_term_effect_info and agent.informed_smart_meter == 1:
                agent.informed_smart_meter = 0
        # set informed state to 1 for selected consumers 
        for agent in nonshifter_sample:
            agent.informed_smart_meter = 1  
            agent.informed_smart_meter_time = self.count
        informed_sm = [agent for agent in self.schedule.agents if agent.informed_smart_meter == 1]
        self.informed_sm =len(informed_sm)
    
    
    def environment_score(self):
        '''Calculate maximum and mean environment score before end of switch period'''
        rounds = int((len(green_list)-1) / self.switch_period)
        values = list(range(1,rounds+1))
        switch_times = [(x * self.switch_period)-1 for x in values]   
        
        if self.count in switch_times:
            agents_env = [sum(list(agent.env_score_prev_period.queue)) for agent in self.schedule.agents]
            self.max_env_score = max(agents_env) # find highest environmental score 
            total_env = sum(agents_env)
            self.mean_env_score = total_env / self.num_agents # calculate mean environmental score

        return self.mean_env_score, self.max_env_score
    
    
    def storage_update(self):
        '''Storage provides energy when there is still shortage after demand shifting''' 
        # battery is charged in times of supply surplus
        if self.production_total > self.total_demand:
            self.storage_soc = min(self.storage_soc + (self.production_total - self.total_demand), self.storage_capacity)
         
        # determine new shortage after switching in current time step, the real mismatch 
        self.real_mismatch_total = self.production_total - self.total_demand
        
        # update SoC in case of shortage 
        if self.real_mismatch_total < 0: # shortage
            self.storage_soc = self.storage_soc + self.real_mismatch_total #mismatch is negative
            self.energy_shortage = - self.real_mismatch_total
        else:
            self.energy_shortage = 0
        
        return self.storage_soc, self.energy_shortage, self.real_mismatch_total


    def price_calc(self):
        '''Update price depending on the policy''' 
        # calculate shortage based on original demand and supply data (shifting excluded)
        self.mismatch_total = (green_list[self.count+1] + fossil_list[self.count+1]) * self.num_agents - (demand_list[self.count+1] * self.num_agents)
        # real-time pricing based on supply
        if self.RTP_policy == 'RTP_Supply': 
            if self.production_total > (self.generation_mean * self.num_agents):  # a lot of supply
                self.price = self.price_flat - self.alpha * (self.production_total / self.num_agents) + self.beta
                return self.price
            elif self.production_total == (self.generation_mean * self.num_agents): # supply is equal to demand
                self.price = self.price_flat
                return self.price
            else:  # little supply
                self.price = self.price_flat + self.alpha * (self.production_total / self.num_agents)  + self.beta
                return self.price
        # real-time pricing based on mismatch
        elif self.RTP_policy == 'RTP_Mismatch':
            if self.production_total - (demand_list[self.count] * self.num_agents) > 0:   # positive mismatch, so more supply than demand
                self.price = self.price_flat - self.gamma * ((self.production_total - (demand_list[self.count] * self.num_agents)) / self.num_agents) + self.beta
                return self.price
            elif self.production_total == (demand_list[self.count] * self.num_agents): # supply is equal to demand
                self.price = self.price_flat
                return self.price
            else:  #  negeative mismatch, so this means shortage
                self.price = self.price_flat + (self.gamma * -1 *((self.production_total - (demand_list[self.count] * self.num_agents))) / self.num_agents) + 0.02 + self.beta
                return self.price  
        # policy is None so there is flat pricing
        else: 
            self.price = self.price_flat
            return self.price
        
        return self.mismatch_total
        
    
    def calc_cost_difference_period(self):
        '''Calculate electricty cost differences between consumers before end of switch period'''
        rounds = int((len(green_list)-1) / self.switch_period)
        values = list(range(1,rounds+1))
        switch_times = [x * self.switch_period -1 for x in values]   
        
        # calculate electricity costs per contract types
        if self.count in switch_times:
            period_semishiftable_cost = sum(self.semishiftable_cost_history[self.count-self.switch_period+1:self.count])
            period_shiftable_cost = sum(self.shiftable_cost_history[self.count-self.switch_period+1:self.count])
            period_no_shift_cost = sum(self.no_shift_cost_history[self.count-self.switch_period+1:self.count])
            
            # compare costs between no shifters and shifters or full shifters
            if self.share_semiandshift > 0:
                self.cost_difference = period_no_shift_cost - period_semishiftable_cost 
            elif self.share_semiandshift == 0 and self.share_shift > 0:
                self.cost_difference = period_no_shift_cost - period_shiftable_cost
            else: # only no shifting contracts
                self.cost_difference = 0
            # in the case that no shifters pay less than shifters, but will not occur with current price settings
            if self.cost_difference < 0:
                self.cost_difference = 0 
             
        return self.cost_difference

################################## model step ################################
            
    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()
        self.green_supply()
        self.fossil_supply()
        self.shifting_population_division()
        self.total_production_calc() 
        self.total_production_predict()
        self.total_mismatch_predict()
        self.demand_total()
        
        rounds = int((len(green_list)-1) / self.switch_period)
        values = list(range(1,rounds+1))
        switch_times = [(x * self.switch_period) for x in values]    
        if self.schedule.time in switch_times:
            self.information_campaign_environment()
            self.information_campaign_smart_meter()
        
        self.environment_score()
        self.storage_update()
        self.price_calc()
        self.price_history.append(self.price)
        self.calc_cost_difference_period()
        self.datacollector.collect(self)
        self.count_steps()
        
    def run_model(self, n):
        for i in range(n):
            self.step()
        