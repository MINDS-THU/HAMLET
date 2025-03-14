A cooperative multi-agent framework modified from [smolagents](https://github.com/huggingface/smolagents/tree/main).

Why choose smolagents as the base code? It manages both calling of tools and agents via python codes, instead of json. It actually makes tool and agent calling in long dialogue look much cleaner and simpler.

To try open_deep_research.app, run
```
python -m examples.open_deep_research.app
```
from the root folder.

TODO list:
- organize tools:
    - essential/default tools (cannot be modified, provide most basic functionalities): search, web browsing, local file reading and writing, converting various file formats to markdown, rag
    - task/domain specific tools (could be created or modified on the fly): operations research related
- automatically building, retrieving and adapting agents for current task

- Compose agent by constructing session-dependent prompt tailored to current task
    - Session-independent
        - System prompt: setting the role
        - Tools available to this agent, matching its role
        - Base model
    - Session-dependent prompt (Need a communication framework to get these information)
        - Main-Task description and progress (optional)
        - what is the bigger picture, i.e. information about the higher-level decisions made to complete the Main-Task and how that is related to the sub-task the current agent is solving
        - Sub-Task description: what sub-task we want this agent to complete now
        - Other context for solving the sub-task - examples:
            - Previous experience in successfully solving similar sub-task
        - Other context for solving the sub-task - feedbacks:
            - Errors + reflection
            - Information acquired from other agents (e.g. asking for necessary information to solve the task)
- improve write_memory_to_messages() method to simplify history for the sake of token cost
- agent communication network beyond one manager + multiple workers? not needed for the moment