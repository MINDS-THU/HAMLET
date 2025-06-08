# Introduction

Combinatorial optimization (CO) occupies a fundamental place in applied mathematics and computer science, seeking optimal solutions within discrete, typically finite, configuration spaces. Formally, a combinatorial optimization problem can be defined as follows: given a finite set $S$ of feasible solutions and an objective function $f : S \rightarrow \mathbb{R}$, the aim is to
\[
s^* = \arg\min_{s \in S} f(s)\quad \text{or}\quad s^* = \arg\max_{s \in S} f(s),
\]
depending on whether $f$ is to be minimized or maximized. CO problems pervade real-world domains, appearing in logistics (e.g., vehicle routing, task scheduling), telecommunications (network design, routing), resource allocation, and task assignment, among many others [1, 2]. These problems are often NP-hard, making exact methods computationally prohibitive for large instances and motivating the search for effective approximation techniques.

Recent advances in deep reinforcement learning (DRL) have opened promising new avenues for tackling such challenging combinatorial problems. DRL integrates reinforcement learning¡ªa paradigm wherein an agent learns to make a sequence of decisions by maximizing cumulative rewards¡ªwith the expressive power of deep neural networks. In the context of CO, the problem is typically framed as a Markov Decision Process (MDP), where states represent partial solutions, actions correspond to solution construction steps, and the reward reflects solution quality. The agent, parameterized by a deep neural network, learns policies $\pi_\theta$ or value functions $V_\theta$ capable of generalizing across problem instances, often surpassing traditional heuristics in both scalability and quality [3, 4].

The intersection of DRL and CO has witnessed a surge of research interest, propelled by DRL's demonstrated success on complex, high-dimensional decision-making tasks. Notable trends include the emergence of neural combinatorial optimization methods, graph-based DRL approaches, and transfer learning strategies to adapt to varying CO landscapes. This rapidly evolving synergy promises not only methodological advances, but also the potential for translating theoretical insights into practical, high-performance solvers for real-world CO problems [3, 4, 5].

This survey is organized as follows. We begin by reviewing foundational concepts in combinatorial optimization and reinforcement learning, then formalize the intersection of these domains. Next, we catalog state-of-the-art DRL methods tailored for CO, describe benchmark problems and evaluation protocols, and analyze empirical results. We conclude by discussing open challenges and prospective research directions.

References


[1] Papadimitriou, C.H., and Steiglitz, K. Combinatorial Optimization: Algorithms and Complexity. Dover, 1998.

[2] Cook, W.J., Cunningham, W.H., Pulleyblank, W.R., and Schrijver, A. Combinatorial Optimization. Wiley, 1998.

[3] Bello, I., Pham, H., Le, Q.V., Norouzi, M., and Bengio, S. Neural Combinatorial Optimization with Reinforcement Learning. arXiv:1611.09940, 2016.

[4] Khalil, E.B., Dai, H., Zhang, Y., Dilkina, B., and Song, L. Learning Combinatorial Optimization Algorithms over Graphs. NeurIPS, 2017.

[5] Joshi, C.K., Cappart, Q., Rousseau, L.-M., Laurent, T., and Bresson, X. Learning TSP Requires Rethinking Generalization. ICLR, 2022.
