from langfuse import Langfuse
import json
from datetime import datetime
import os
import time
from collections import defaultdict
from langfuse.api.core.api_error import ApiError

# Robust API wrapper
def robust_api_call(func, *args, sleep_between_calls=True, **kwargs):
    max_retries = 5
    base_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)

            if sleep_between_calls:
                time.sleep(0.67)  # Throttle to ~90 req/min

            return result
        except ApiError as e:
            if e.status_code == 429:
                retry_after = 60
                if 'Retry-After' in str(e.body):
                    try:
                        retry_after = int(str(e.body).split("Retry-After: ")[1].split()[0])
                    except Exception:
                        pass
                else:
                    retry_after = base_delay * (2 ** attempt)

                print(f"Rate limit hit. Waiting for {retry_after} seconds before retrying...")
                time.sleep(retry_after)
            else:
                raise
    raise RuntimeError("Max retries exceeded for API call")

# Helper functions
def get_span_kind(obs):
    return (obs.metadata or {}).get("attributes", {}).get("openinference.span.kind")

def get_agent_name(obs):
    return (obs.metadata or {}).get("attributes", {}).get("openinference.agent.name")

def get_tool_name(obs):
    return (obs.metadata or {}).get("attributes", {}).get("openinference.tool.name")

def process_observations(observations):
    obs_by_id = {obs.id: obs for obs in observations}
    children_map = defaultdict(list)
    root_ids = set(obs_by_id)

    for obs in observations:
        pid = obs.parent_observation_id
        if pid and pid in obs_by_id:
            children_map[pid].append(obs.id)
            root_ids.discard(obs.id)

    agent_history = defaultdict(list)

    def dfs(obs_id, current_agent=None, current_tool=None):
        obs = obs_by_id[obs_id]
        span_kind = get_span_kind(obs)

        if span_kind == "AGENT":
            current_agent = get_agent_name(obs)
        if span_kind == "LLM" and current_agent:
            agent_history[current_agent].append({
                "prompt": format_input(obs.input),
                "completion": format_output(obs.output)
            })

        for child_id in children_map.get(obs_id, []):
            dfs(child_id, current_agent, current_tool)

    for root_id in root_ids:
        dfs(root_id)
    return agent_history

def format_input(obs_input: dict):
    input_list = obs_input["messages"]
    res_list = []
    for i in range(len(input_list)):
        if input_list[i]["role"] == "tool-call":
            continue
        if input_list[i]["role"] == "system":
            res_list.append({
                "role": "system",
                "content": input_list[i]["content"][0]["text"]
            })
        elif input_list[i]["role"] in ["user", "tool-response"]:
            res_list.append({
                "role": "user",
                "content": input_list[i]["content"][0]["text"]
            })
        elif input_list[i]["role"] == "assistant":
            res_list.append({
                "role": "assistant",
                "content": input_list[i]["content"][0]["text"]
            })
    del obs_input
    return res_list

def format_output(obs_output: dict):
    return {"role": obs_output["role"], "content": obs_output["content"]}

def process_session(session, save_dir):
    session_role_data_dict = {}
    print(f"\nSession ID: {session.id}")
    if session.id not in session_role_data_dict:
        session_role_data_dict[session.id] = {}

    traces_resp = robust_api_call(langfuse.fetch_traces, session_id=session.id)
    traces = traces_resp.data
    print(f"Found {len(traces)} traces.")

    for trace in traces:
        print(f"Trace ID: {trace.id}")
        full_trace = robust_api_call(langfuse.fetch_trace, trace.id)
        observations = full_trace.data.observations
        agent_history = process_observations(observations)

        for agent, history in agent_history.items():
            print(f"Agent: {agent}")
            if agent not in session_role_data_dict[session.id]:
                session_role_data_dict[session.id][agent] = []
            session_role_data_dict[session.id][agent].extend(history)

    # Save
    for session_id, role_data in session_role_data_dict.items():
        for role, data in role_data.items():
            session_dir = f"{save_dir}/{session_id}"
            os.makedirs(session_dir, exist_ok=True)
            with open(f"{session_dir}/{role}.json", "w") as f:
                json.dump(data, f, indent=4)

if __name__ == "__main__":
    # Initialize Langfuse client
    # Assuming that you have set up your langfuse cloud account and have created an existing project saving the data,
    # then you can use the following code to fetch and process sessions saved under this project.
    # Make sure to replace the secret_key and public_key with your actual keys.
    # this will save the data in a directory called "finetuning_dataset" with the following structure:
    # finetuning_dataset/
    #   <session_id>/
    #       <agent_name>.json
    # each <agent_name>.json file will contain the conversation history for that agent in the session.

    langfuse = Langfuse(
        secret_key="",
        public_key="",
        host="https://cloud.langfuse.com",
    )

    for page_num in range(10): # this fetches the first 10 pages of sessions
        print(f"Fetching sessions for page {page_num}...")
        sessions = robust_api_call(langfuse.fetch_sessions, page=page_num).data
        print(f"Found {len(sessions)} sessions")

        for session in sessions:
            session_dir = os.path.join("finetuning_dataset", session.id)

            # Skip if already processed
            if os.path.exists(session_dir):
                print(f"Skipping session {session.id} (already processed).")
                continue

            process_session(session, "finetuning_dataset")