
import numpy as np

# 1. Forbid any arc connected to supply node 1
constraint_1 = "We want to exclude any connections coming from supply node 1. This could be due to reliability concerns, maintenance schedules, or other priorities, so let’s make sure no demand point is directly linked to supply node 1."
def check_constraint_1(network_design):
    return np.any(network_design[1, :] == 1)

# 2. Investment budget with variable arc costs
constraint_2 = "We are working with a budget of $400. Each connection to demand point 3 costs $150, while any other connection costs $100. Let us design the network to maximize expected profits without exceeding this budget. Prioritize cost-effective connections that give the best returns within our spending limit."
def check_constraint_2(network_design, budget=400):
    cost = 0
    for i in range(network_design.shape[0]):
        for j in range(network_design.shape[1]):
            if network_design[i, j] == 1:
                cost += 150 if j == 3 else 100
    return cost > budget

# 3. No node has more than 2 arcs
constraint_3 = "For simplicity and manageability, ensure that each node – whether supply or demand – has no more than two connections. This helps avoid overloading and keeps the network balanced. Adjust the design to make sure we’re not exceeding two connections at any one point."
def check_constraint_3(network_design):
    rows_sum = np.sum(network_design, axis=1)
    cols_sum = np.sum(network_design, axis=0)
    return np.any(rows_sum > 2) or np.any(cols_sum > 2)

# 4. No demand node has more than 2 arcs
constraint_4 = "Let’s avoid connecting any demand point to more than two supply nodes. This helps manage logistical challenges at the demand sites and simplifies coordination. Make sure each demand node is supplied by no more than two sources."
def check_constraint_4(network_design):
    return np.any(np.sum(network_design, axis=0) > 2)

# 5. No supply node has more than 2 arcs
constraint_5 = "Each supply node should also be limited to two outgoing connections. This can help prevent bottlenecks at the supply points and ensure we’re not overextending any one supply node. Adjust connections so each supply point only serves up to two demand nodes."
def check_constraint_5(network_design):
    return np.any(np.sum(network_design, axis=1) > 2)

# 6. Mutual exclusivity between supply nodes 0 and 1
constraint_6 = "If we use supply node 0, then we won’t connect supply node 1 to any demand nodes, and vice versa. Let’s choose one of these supply nodes for connections but not both. This reduces redundancy and helps streamline operations in areas where supply nodes may cover overlapping regions."
def check_constraint_6(network_design):
    has_connection_0 = np.any(network_design[0, :] == 1)
    has_connection_1 = np.any(network_design[1, :] == 1)
    return has_connection_0 and has_connection_1

# 7. If more than 2 arcs stem from node 0, no arcs from nodes 1 or 2
constraint_7 = "If we add more than two connections from supply node 0, then let’s skip adding any connections from nodes 1 or 2. This approach avoids unnecessary overlap and keeps the system efficient. Adjust the design to limit nodes 1 and 2 if we’re heavily relying on node 0."
def check_constraint_7(network_design):
    has_0_more_than_2 = np.sum(network_design[0, :] == 1) > 2
    has_connection_1 = np.any(network_design[1, :] == 1)
    has_connection_2 = np.any(network_design[2, :] == 1)
    return has_0_more_than_2 and (has_connection_1 or has_connection_2)

# 8. Nodes 0 and 2 have no more than 2 arcs each, with total of 7 arcs
constraint_8 = "We’ll set a total of 7 connections, but to keep things balanced, don’t let nodes 0 and 2 have more than two connections each. This helps maintain an even distribution of load across the network. Check that our final design respects this cap on connections for nodes 0 and 2."
def check_constraint_8(network_design, max_arcs=7):
    total_arcs = np.sum(network_design)
    arcs_0 = np.sum(network_design[0, :] == 1)
    arcs_2 = np.sum(network_design[2, :] == 1)
    return total_arcs != max_arcs or arcs_0 > 2 or arcs_2 > 2

# 9. At least 2 arcs stem from node 4
constraint_9 = "Supply node 4 is critical for meeting demand, so it needs a minimum of two outgoing connections. Adjust the design to ensure node 4 meets this requirement within our total connection limit."
def check_constraint_9(network_design, required_arcs_from_4=2, total_arcs=5):
    arcs_from_4 = np.sum(network_design[4, :] == 1)
    total_connections = np.sum(network_design)
    return total_connections != total_arcs or arcs_from_4 < required_arcs_from_4

# 10. Arcs from nodes 0 and 4 to node 2 cannot both exist
constraint_10 = "We want to prevent both supply nodes 0 and 4 from connecting to demand node 2 at the same time. If node 0 connects to node 2, skip any connection from node 4 to node 2, and vice versa. This minimizes overlap and avoids redundancy."
def check_constraint_10(network_design):
    arc_0_to_2 = network_design[0, 2] == 1
    arc_4_to_2 = network_design[4, 2] == 1
    return arc_0_to_2 and arc_4_to_2
