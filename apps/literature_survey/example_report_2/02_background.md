# 2. Background

## Combinatorial Optimization: Definitions and Classical Approaches

Combinatorial optimization (CO) is the task of finding an optimal object, arrangement, or selection from a finite but typically large set of possibilities, subject to constraints. Formally, given a finite set $S$ and an objective function $f: S \rightarrow \mathbb{R}$, the goal is to find $x^* \in S$ such that
\[
f(x^*) = \min_{x \in S} f(x)
\]
or $\max_{x \in S} f(x)$ for maximization problems [1].

**Key Combinatorial Optimization Problems:**

- **Traveling Salesman Problem (TSP):**
  Given a set of $n$ cities and a distance matrix $D \in \mathbb{R}^{n \times n}$, find the shortest possible tour that visits each city exactly once and returns to the origin city. This can be formulated as finding a permutation $\pi$ of the cities that minimizes:
  \[
  \min_{\pi \in \text{Perm}(n)} \sum_{i=1}^{n} D_{\pi(i),\pi(i+1)}
  \]
  where $\pi(n+1) = \pi(1)$ [2].

- **Vehicle Routing Problem (VRP):**
  Given a depot, a set of customers with associated demands, and a fleet of vehicles with limited capacity, determine optimal routes for the vehicles such that all customer demands are satisfied with minimum routing cost, and operational constraints (e.g., vehicle capacity, route length) are not violated [3].

- **Graph Coloring Problem:**
  Assign colors to the vertices of a graph $G = (V, E)$ such that no two adjacent vertices share the same color, using the minimum number of colors possible. Formally: $c : V \rightarrow \{1, ..., k\}$ where $c(u) \neq c(v)$ for every edge $(u, v) \in E$ [4].

- **Job-Shop Scheduling Problem (JSP):**
  Given a set of jobs, each with an ordered sequence of operations requiring specific machines for given durations, the goal is to schedule all operations on the machines (each can process only one job at a time) to minimize the makespan (the total completion time) [5].

**Classical Approaches:**

- **Exact Algorithms:**
  - *Branch-and-Bound:* Systematically explores parts of the solution space, using bounds to prune suboptimal regions and guarantee optimality [6].
  - *Dynamic Programming:* Decomposes problems into overlapping subproblems, solving and storing each to avoid redundant calculations, suitable for problems with optimal substructure [7].

- **Heuristics and Metaheuristics:**
  - *Greedy Algorithms:* Construct solutions incrementally, making the locally optimal choice at each step [8].
  - *Local Search:* Begins with an initial solution and iteratively improves it by exploring local neighbors [9].
  - *Simulated Annealing:* A stochastic local search that allows occasional moves to worse solutions to escape local optima, with a temperature parameter controlling the probability [10].
  - *Genetic Algorithms:* Population-based metaheuristics inspired by biological evolution, using operations like selection, crossover, and mutation to evolve better solutions over generations [11].

#### References


[1] Combinatorial optimization. [Wikipedia](https://en.wikipedia.org/wiki/Combinatorial_optimization), accessed 2024.

[2] Travelling salesman problem. [Wikipedia](https://en.wikipedia.org/wiki/Travelling_salesman_problem), accessed 2024.

[3] P. Toth & D. Vigo (Eds.), *The Vehicle Routing Problem.* SIAM Monographs on Discrete Mathematics and Applications, 2002.

[4] Graph coloring. [Wikipedia](https://en.wikipedia.org/wiki/Graph_coloring), accessed 2024.

[5] Job-shop scheduling. [Wikipedia](https://en.wikipedia.org/wiki/Job-shop_scheduling), accessed 2024.

[6] Pardalos, P. M., & Resende, M. G. C. (Eds.). "Branch-and-Bound Algorithms." In *Handbook of Applied Optimization*. Oxford University Press, 2002.

[7] Bellman, R. "Dynamic Programming." *Science*, 153(3731), 34-37, 1966.

[8] Feo, T. A., & Resende, M. G. C. "Greedy Randomized Adaptive Search Procedures." *Journal of Global Optimization*, 6, 109-133, 1995.

[9] Russell, S., & Norvig, P. *Artificial Intelligence: A Modern Approach* (4th ed.), Pearson, 2020.

[10] Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P., "Optimization by Simulated Annealing." *Science*, 220(4598), 671-680, 1983.

[11] Holland, J. H. *Adaptation in Natural and Artificial Systems*. University of Michigan Press, 1975.


## Reinforcement Learning and Deep RL

A Markov Decision Process (MDP) provides the formal foundation for reinforcement learning, modeling sequential decision problems as a tuple $(S, A, P, R, \gamma)$:
- $S$ is the set of states,
- $A$ is the set of actions,
- $P: S \times A \times S \rightarrow [0,1]$ is the transition probability function $P(s'|s,a)$,
- $R: S \times A \rightarrow \mathbb{R}$ is the reward function,
- $\gamma \in [0,1]$ is the discount factor [12].

**Reinforcement Learning Objective and Elements:**

The objective in RL is to learn a policy $\pi$ that maximizes the expected cumulative reward (return) for an agent interacting with an environment modeled as an MDP. A (stochastic) policy $\pi(a|s)$ defines the probability of taking action $a$ in state $s$ [13].

- *Value Function*: The state-value function under policy $\pi$, $V^{\pi}(s)$, is the expected return starting from state $s$:
  \[
  V^{\pi}(s) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0 = s \right]
  \]

- *Action-Value Function*: $Q^{\pi}(s,a) = \mathbb{E}_{\pi}[ \sum_{t=0}^\infty \gamma^t r_{t+1} \mid s_0 = s, a_0 = a ]$

- *Bellman Equation*: The Bellman equation for $V^{\pi}$:
  \[
  V^{\pi}(s) = \sum_{a \in A} \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'\in S} P(s'|s,a) V^{\pi}(s') \right]
  \]

**Deep Reinforcement Learning (Deep RL):**

Deep RL applies deep neural networks to approximate value functions or policies, enabling learning in high-dimensional or unstructured spaces. Key families include:

- **Deep Q-Networks (DQN):** Use neural networks to approximate the action-value (Q) function. DQN selects actions by maximizing the learned Q-values and incorporates techniques like experience replay and target networks for stability [14].

- **Policy Gradients:** Directly parameterize the policy and update its parameters by gradient ascent on expected reward. Useful for high-dimensional or continuous action spaces [15].

- **Actor-Critic:** Combines value-based and policy-based ideas. The "actor" learns the policy and the "critic" estimates value functions, with the critic guiding the actor for more efficient and stable learning [16].

**Intuition:**
- DQN estimates how good each action is (in terms of future rewards) and chooses accordingly.
- Policy gradients learn a probability distribution over actions directly, updating towards more rewarding actions.
- Actor-critic combines both, enabling scalable and stable learning in complex environments.

#### References


[12] Puterman, M. L., *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley, 1994.

[13] Sutton, R. S., & Barto, A. G., *Reinforcement Learning: An Introduction* (2nd ed.), MIT Press, 2018.

[14] Mnih, V., et al. "Human-level control through deep reinforcement learning." *Nature* 518, 529¨C533, 2015.

[15] Sutton, R. S., et al., "Policy Gradient Methods for Reinforcement Learning with Function Approximation," NIPS, 2000.

[16] Konda, V. R., & Tsitsiklis, J. N., "Actor-Critic Algorithms," NIPS, 2000.

