# 3. Formulating Combinatorial Optimization as Reinforcement Learning Problems

Combinatorial optimization (CO) problems¡ªwhere the goal is to minimize or maximize an objective function $f(x)$ over a discrete set of candidates $x \in \mathcal{X}$¡ªare classically solved using algorithmic or mathematical programming techniques. Recent research has shown that framing these problems as sequential decision-making processes allows them to be tackled using deep reinforcement learning (RL) methods [1][2][3][4][5]. This section describes how CO problems are mapped to Markov Decision Processes (MDPs) suitable for RL, discusses challenges in this translation, and provides a detailed example with the Traveling Salesman Problem (TSP).

## Markov Decision Process Formulation for CO

An MDP for combinatorial optimization is defined by the tuple $(\mathcal{S}, \mathcal{A}, P, R, s_0, T)$, where:

- **State space ($\mathcal{S}$):** Each state $s \in \mathcal{S}$ encodes the partial solution constructed so far, as well as additional information about the problem instance (e.g., which elements have been selected or assigned).
- **Action space ($\mathcal{A}$):** Actions correspond to ways of extending or modifying the current partial solution, such as assigning the next variable, selecting an element, or choosing how to complete a structure.
- **Transition function ($P$):** Given the current state $s_t$ and action $a_t$, the environment deterministically (or stochastically, if the problem has randomness) transitions to the next state $s_{t+1}$. For most CO problems, transitions are deterministic and reflect adding an element to the solution.
- **Reward function ($R$):** The reward provides feedback about solution quality. It is often **sparse** (available only at the end of the episode¡ªi.e., when a complete solution is built), but can sometimes be designed as **stepwise** (e.g., incremental improvement or penalty at each step). For minimization, negative costs are standard.
- **Start state ($s_0$):** The initial state corresponds to having constructed no part of the solution (e.g., empty selection, or just the initial node).
- **Termination ($T$):** Episodes terminate when a complete, valid solution is constructed¡ªthat is, when the stopping criteria of the CO problem are met.

## Translating CO Problems into MDPs: Key Considerations

When mapping a CO problem to an RL/MDP framework, several criteria must be considered:

- **State Representation:** How to encode partial solutions such that they contain sufficient information for the agent to make optimal decisions. For example, in routing problems, the state might include the sequence of nodes already visited and the remaining cities.
- **Action Design:** Ensuring that the action space is well-defined (e.g., only feasible choices), and scalable for large problems.
- **Reward Assignment:** Choosing between sparse rewards (assigned at episode end) or potential-shaping with intermediate rewards. The choice impacts agent learning efficiency.
- **Handling Constraints:** Many CO problems have hard constraints. These can be enforced through state/action restrictions or by penalizing infeasible solutions in the reward function [2][3][5].
- **Partial vs. Complete Solutions:** The RL environment typically builds the solution step by step, so the agent's experience at each state is closely linked to the evolving partial solution [2][3].

## Worked Example: The Traveling Salesman Problem as an RL/MDP

**Problem Statement:** Given $N$ cities and a distance matrix $D = [d(i,j)]$, find a tour (a permutation $\pi$ of the cities) that minimizes the total travel distance:
\[
    \min_{\pi \in S_N} \Bigg( d(\pi_1, \pi_2) + d(\pi_2, \pi_3) + \ldots + d(\pi_N, \pi_1) \Bigg)
\]
where $S_N$ is the set of all permutations of $N$ elements.

Formulated as an RL/MDP problem:

- **State ($s_t$):**
  At step $t$, the state is defined as $s_t = (c_t, V_t)$, where $c_t$ is the current city and $V_t$ is the set of visited cities up to step $t$.

- **Action ($a_t$):**
  At each step, the action $a_t$ is the choice of the next unvisited city to visit, i.e., $a_t \in \{1,\ldots,N\} \setminus V_t$.

- **Transition:**
  Taking action $a_t$ from state $s_t$ moves to $s_{t+1} = (a_t, V_t \cup \{a_t\})$.

- **Reward ($r_t$):**
  The reward at step $t$ is typically the negative distance traveled:
\[
    r_t = -d(c_t, a_t)
\]
  Optionally, a terminal reward can be given at episode end for returning to the starting city, i.e., $r_T = -d(c_T, c_0)$.

- **Episode Termination:**
  The episode terminates when all cities have been visited: $|V_t| = N$.

- **Objective:**
  Find a policy $\pi$ that maximizes the expected cumulative reward (i.e., minimizes the total tour length):
\[
    \max_{\pi} \; \mathbb{E}_{\pi}\Bigg[\sum_{t=0}^{N} r_t\Bigg]
\]
where the expectation is with respect to the agent's policy over possible tours [1][2][3].

This mapping enables the use of RL algorithms, such as policy gradient methods, to train agents that incrementally build valid tours while seeking to minimize total travel distance [1][2].

## References


[1] Irwan Bello, Hieu Pham, Quoc V. Le, Mohammad Norouzi, Samy Bengio. Neural Combinatorial Optimization with Reinforcement Learning. arXiv:1611.09940, 2016.  [https://arxiv.org/abs/1611.09940](https://arxiv.org/abs/1611.09940)

[2] Nathan Grinsztajn. Reinforcement learning for combinatorial optimization. PhD Thesis, Universit¨¦ Paris Dauphine¨CPSL, 2023. [https://theses.hal.science/tel-04353766v1](https://theses.hal.science/tel-04353766v1)

[3] Y. Liu, X. Li, J. Wang. A Review of Research on Reinforcement Learning for Solving Combinatorial Optimization Problems. CAAI Artificial Intelligence Research, 2023. [https://www.sciopen.com/article/10.13568/j.cnki.651094.651316.2023.02.02.0001](https://www.sciopen.com/article/10.13568/j.cnki.651094.651316.2023.02.02.0001)

[4] Anna Kuzin, Karthik Narasimhan. Reinforcement Learning with Combinatorial Actions. NeurIPS 2020. [https://papers.nips.cc/paper/2020/file/06a9d51e04213572ef0720dd27a84792-Paper.pdf](https://papers.nips.cc/paper/2020/file/06a9d51e04213572ef0720dd27a84792-Paper.pdf)

[5] Haoran Chen, Rui Song, Weinan Zhang. Reinforcement learning for combinatorial optimization: A survey. Computers & Operations Research, 2021. [https://www.sciencedirect.com/science/article/abs/pii/S0305054821001660](https://www.sciencedirect.com/science/article/abs/pii/S0305054821001660)

