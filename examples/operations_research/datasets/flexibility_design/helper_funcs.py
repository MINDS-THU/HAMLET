import numpy as np
from copy import deepcopy
import json
from gurobipy import *

class FlexSecondStage:
    # initialize a model by setting variables and constraints (with RHS values)
    def __init__(self, name, demand_vec, capacity_vec, profit_mat, design):
        """
        :param name: Model name
        :param demand_vec: 2d vector where each column contains the demand vector of a scenario.
        :param capacity_vec: 2d vector where each column contains the capacity vector of a scenario.
        :param profit_mat: 2d matrix with (i,j)th entry containing the unit profit of matching supply i to demand j
        :param design: 2d binary matrix describing the network configuration design
        """
        self.m = Model(name)
        self.f = {}
        self.cs = {}
        self.ct = {};  # f is the flow variable, b is the lost sales variable, cs is the (supply) plant constraints,
        # ct is the (demand) product constraints
        self.ub = {}
        self.num_demand = len(demand_vec)
        self.num_capacity = len(capacity_vec)
        self.demand_scenario_vec = demand_vec
        self.capacity_scenario_vec = capacity_vec

        for j in range(self.num_demand):
            for i in range(self.num_capacity):
                self.f[i, j] = self.m.addVar(name='f_%s' % i + '%s' % j, ub=design[i, j] * 10e30)
                self.f[i, j].setAttr(GRB.attr.Obj, profit_mat[i, j])
        self.m.update()

        for i in range(self.num_capacity):
            self.cs[i] = self.m.addConstr(quicksum(self.f[i, j] for j in range(self.num_demand)) <= 0, name='cs_%s' % i)
        for j in range(self.num_demand):
            self.ct[j] = self.m.addConstr(quicksum(self.f[i, j] for i in range(self.num_capacity)) <= 0,
                                          name='ct_%s' % j)
        self.m.update()
        self.m.setAttr("ModelSense", GRB.MAXIMIZE)
        self.m.setParam('OutputFlag', 0)

    def solve_scenario(self, scenario_index):
        for j in range(self.num_demand):
            self.ct[j].setAttr(GRB.attr.RHS, self.demand_scenario_vec[j, scenario_index])
        for i in range(self.num_capacity):
            self.cs[i].setAttr(GRB.attr.RHS, self.capacity_scenario_vec[i, scenario_index])
        self.m.optimize()
        return self.m.objVal

def greedy(initial_design, capacity_vec, demand_vec, profit_mat):
    """
    :param initial_design: numpy 2d array for the initial network structure
    :param capacity_vec: numpy 2d array consists of capacities in all scenarios
    :param demand_vec: numpy 2d array consists of demands in all scenarios
    :param profit_mat: numpy 2d array describing the unit profit on each supply-demand node pair
    :param fixed_costs: numpy 2d array describing the fixed cost on installing arc on each supply-demand node pair
    :return: finds the arc that can be added to initial_design which has the largest improvement to the expected profit. Return both the arc and the improvement.
    """
    _, num_samples = capacity_vec.shape
    best_profit = -1e10
    X = initial_design.copy()
    it_arcs = np.nditer(X, flags=['multi_index'])
    arc_index = {}

    while not it_arcs.finished:
        cur_arc = it_arcs.multi_index

        if X[cur_arc[0], cur_arc[1]] == 0:
            samp_profit = np.zeros(num_samples)
            # Set up the basic structure of the second stage problem
            secondstage_lp = FlexSecondStage("LP", demand_vec, capacity_vec, profit_mat, initial_design)
            secondstage_lp.f[cur_arc[0], cur_arc[1]].setAttr(GRB.attr.UB, 10e30)
            for s in range(num_samples):
                samp_profit[s] = secondstage_lp.solve_scenario(s)
            secondstage_lp.f[cur_arc[0], cur_arc[1]].setAttr(GRB.attr.UB, 0)
            new_profit = np.average(samp_profit)
            
            print("cur arc {}, profit {}".format(cur_arc, new_profit))
            if (new_profit > best_profit):
                best_profit = new_profit
                arc_index = it_arcs.multi_index
        it_arcs.iternext()

    return arc_index, best_profit

def greedy_solution(m, n, capacity, mean_d, std_d, K, P, initial_network, n_samples=1000):
    final_network = deepcopy(initial_network)
    ## sample capacity and demand ##
    sampled_capacity = np.zeros((m, n_samples))
    # np.random.seed(0)
    for i in range(n_samples):
        # capacity = np.random.normal(self.mean_c, self.std_c)
        # # truncate demand at two standard deviations
        # capacity = np.maximum(capacity, 0)
        # capacity = np.maximum(capacity, self.mean_c - 2 * self.std_c)
        # capacity = np.minimum(capacity, self.mean_c + 2 * self.std_c)
        sampled_capacity[:, i] = capacity

    sampled_demand = np.zeros((n, n_samples))
    # np.random.seed(1)
    for i in range(n_samples):
        demand = np.random.normal(mean_d, std_d)
        # truncate demand at two standard deviations
        demand = np.maximum(demand, 0)
        demand = np.maximum(demand, mean_d - 2 * std_d)
        demand = np.minimum(demand, mean_d + 2 * std_d)
        sampled_demand[:, i] = demand

    for k in range(K):
        arc_ind, best_profit = greedy(final_network, sampled_capacity, sampled_demand, P)
        print("add arc ind: {}".format(arc_ind))
        # print("add arc ind: {}".format(arc_ind))
        final_network[arc_ind] = 1
    return final_network, best_profit
