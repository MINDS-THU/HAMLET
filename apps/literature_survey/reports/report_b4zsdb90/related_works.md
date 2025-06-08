# Related Works

Finetuning large language models (LLMs) for improved bargaining ability, specifically on data generated via simulation, is an emerging research frontier at the intersection of natural language processing (NLP), multi-agent learning, and game-theoretic negotiation. This section reviews and synthesizes supporting literature, particularly focusing on recent advances relevant to LLM finetuning, simulation/synthetic data for training, and benchmarks for negotiation and bargaining.

## LLM Finetuning for Negotiation and Bargaining

Recent works have established that LLMs can be taught negotiation and bargaining skills through finetuning using supervised or reinforcement learning techniques, often with simulated or real datasets.

- **FishBargain** proposes a modular LLM-powered agent for online fleamarket sellers, employing a three-stage pipeline (price extractor, policy planner, utterance generator) to optimize automated price bargaining. While not centered on explicit finetuning, the system relies on domain-specific adaptation of LLMs for decision and dialog generation, demonstrating concrete efficacy in seller negotiation tasks [1].

- **AgreeMate** introduces an explicit finetuning pipeline, leveraging prompt engineering and supervised learning to teach LLMs to haggle. The work examines LLMs' strategic negotiation capacities traversing multiple roles (buyer/seller) and provides detailed analysis of learning outcomes following both prompt-based and finetuned approaches [2].

- **Lewis et al. (Deal or No Deal?)** present an early approach to training negotiation agents via end-to-end neural learning, using supervised and reinforcement learning plus self-play on a negotiation dataset. This classic work formalizes the negotiation dialogue paradigm for end-to-end LLMs, including reward-based optimization without annotated dialogue states [7].

- **Measuring Bargaining Abilities of LLMs** introduces a benchmark and a buyer-enhancement method to improve LLMs' bargaining, explicitly framing bargaining as an asymmetric negotiation and fine-tuning LLMs for role-specific enhancements using both prompting and agent interaction [9].

## Synthetic Data and Simulation in Negotiation Training

Simulation and self-play are critical methodologies enabling scalable generation of negotiation data, especially where real annotated data is rare or costly. Multiple works substantiate the use of simulation in training and evaluating bargaining agents:

- **Improving Language Model Negotiation with Self-Play and In-Context Learning from AI Feedback** demonstrates that LLMs improve at negotiation via repeated self-play, reflection, and AI-feedback. The iterative process allows models to refine strategy autonomously, leveraging their own outputs as a simulation corpus for subsequent in-context learning [3].

- **LLM Agents for Bargaining with Utility-based Feedback** proposes utility-based feedback to guide bargaining agents, introducing the BargainArena simulated benchmark with complex negotiation environments (e.g., deception, monopoly). The study empirically shows that structuring simulation around utility optimization, and feeding back utility-aligned signals, sharpens LLM bargaining abilities [4].

- **LLMs at the Bargaining Table** explores the convergence of LLM agent negotiations to game-theoretic equilibria (Rubinstein's bargaining solution [8]), simulating repeated agent-agent bargaining and examining matches with theoretical predictions [6].

- **Lewis et al. (Deal or No Deal?)** harnesses simulation and self-play for end-to-end agent learning; models improve negotiation efficacy by interacting with simulated counterparts, mirroring real-world negotiation structures [7].

## Benchmarks and Evaluation Platforms for LLM Negotiation

Robust, standardized evaluation of bargaining ability has catalyzed the field. Several works have contributed benchmarks which encapsulate both adversarial and cooperative aspects of negotiation:

- **How Well Can LLMs Negotiate? NEGOTIATIONARENA Platform and Analysis** introduces a reusable, extensible framework (NEGOTIATIONARENA) for systematic evaluation of LLM negotiation skill under controlled, simulated scenarios [5].

- **LLM Agents for Bargaining with Utility-based Feedback** establishes the BargainArena platform specifically for negotiation with utility-relevant settings [4].

- **Measuring Bargaining Abilities of LLMs** creates a formal task/benchmark with real product price data (AmazonHistoryPrice) and proposes concrete scoring for LLM-mediated negotiation [9].

- **Evaluating Language Model Agency through Negotiations** puts forth negotiation games specifically for assessing LM agency in practical settings, emphasizing adversarial and interactive evaluation [8].

## Trends, Gaps, and Positioning

**Trends:**
- Simulation/self-play (incl. agent-agent, AI-driven) is rapidly supplanting fully human-curated data for negotiation training, allowing LLMs to jointly learn language and strategy.
- Utility-based/role-based feedback is emerging as an effective technique for shaping negotiation preferences and outcomes.
- The proliferation of benchmarks and platforms (BargainArena, NEGOTIATIONARENA, Deal or No Deal) enables rigorous, reproducible evaluation for LLM bargaining research.

**Gaps:**
- While several works leverage simulation and feedback for agent learning, few systematically explore the effects of finetuning on *purely* synthetic (i.e., simulation-generated) data or combine this with explicit price/deal bargaining as the focal task.
- Many prior efforts focus on symmetric negotiations or roles; asymmetric, real-world scenarios (like online marketplaces) are less explored at scale.

**Our Contribution** builds upon these directions by focusing on finetuning LLMs exclusively with simulation-generated negotiation data, targeted at the price bargaining problem, and quantifying performance under realistic, asymmetric market-driven scenarios.

## Formal Definitions and Algorithms

**Bargaining Task (Formalization):**
Let the negotiation environment be defined by a tuple $(A, O, r, \mathcal{T})$:

- $A$ ！ Set of agents (e.g., buyer and seller).
- $O$ ！ Space of possible offers (actions at each turn).
- $r$ ！ Reward function mapping final agreed offer to agent utilities.
- $\mathcal{T}$ ！ Turn-based protocol defining agent interaction until agreement or termination.

**Utility-Based Feedback:**
Given agent $i$, utility function $u_i(o)$ assigns a score to offer $o$. Utility-based learning then optimizes agent language/policy such that $\mathbb{E}_{o \sim \text{Policy}}[u_i(o)]$ is maximized (for self-interest) or achieves a negotiated optimum (for Nash/social welfare).

## References


[1] Dexin Kong, Xu Yan, Ming Chen. FishBargain: An LLM-Empowered Bargaining Agent for Online Fleamarket Platform Sellers. arXiv preprint arXiv:2502.10406, 2025. https://arxiv.org/abs/2502.10406

[2] Paul, Rohan, et al. AgreeMate: Teaching LLMs to Haggle. arXiv preprint arXiv:2412.18690, 2024. https://arxiv.org/abs/2412.18690

[3] Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Eric Wallace, Yizhong Wang, Sewon Min, Hannaneh Hajishirzi, and Dan Roth. Improving Language Model Negotiation with Self-Play and In-Context Learning from AI Feedback. arXiv preprint arXiv:2305.10142, 2023. https://arxiv.org/abs/2305.10142

[4] LLM Agents for Bargaining with Utility-based Feedback. arXiv preprint arXiv:2505.22998, 2025. https://arxiv.org/abs/2505.22998

[5] How Well Can LLMs Negotiate? NEGOTIATIONARENA Platform and Analysis. arXiv preprint arXiv:2402.05863, 2024.

[6] LLMs at the Bargaining Table. OpenReview. https://openreview.net/pdf?id=n0RmqncQbU

[7] Lewis, M., Yarats, D., Dauphin, Y., Parikh, D., & Batra, D. Deal or No Deal? End-to-End Learning for Negotiation Dialogues. arXiv preprint arXiv:1706.05125, 2017. https://arxiv.org/abs/1706.05125

[8] Evaluating Language Model Agency through Negotiations. arXiv preprint arXiv:2401.04536, 2024.

[9] Zhuosheng Zhang, et al. Measuring Bargaining Abilities of LLMs: A Benchmark and A Buyer-Enhancement Method. Findings of the Assoc. for Computational Linguistics: ACL 2024. https://aclanthology.org/2024.findings-acl.213.pdf

