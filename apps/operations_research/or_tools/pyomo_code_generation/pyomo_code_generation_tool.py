import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from smolagents import Tool
import generate_prompt

# define SmolAgent tool
class PyomoCodeGeneration(Tool):
    name = "generate_pyomo_code"
    description = """
    This tool generates the five-element description and Pyomo code based on a natural language optimization problem description.
    The input should be a natural language question describing the optimization problem.
    The output will be a string containing the Pyomo code, with five-element description being comments.
    """
    inputs = {
        "problem_description": {
            "type": "string",
            "description": "A natural language description of the optimization problem."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        # === Load model and tokenizer ===
        self.device = "cuda"

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path="ant-opt/LLMOPT-Qwen2.5-14B",
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="ant-opt/LLMOPT-Qwen2.5-14B")

    def _infer_five_elements(self, question: str) -> str:
        """
        Infers the five key elements (objective, variables, constraints, etc.)
        from a natural language optimization problem.
        
        Args:
            question: A natural language question describing the optimization problem.
        
        Returns:
            A string containing the extracted elements.
        """
        messages = [{"role": "user", "content": generate_prompt.Q2F(question)}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=8192)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = response.replace("\\\\n", "\n").replace("&#39;", "'") \
                        .replace("&lt;", "<").replace("&gt;", ">") \
                        .replace("\\\\\"", "\"")

        if "```text" in response:
            return response.split("```text")[1].split("```")[0]
        elif "```plaintext" in response:
            return response.split("```plaintext")[1].split("```")[0]
        elif "```" in response:
            return response.split("```")[1].split("```")[0]
        else:
            return response.strip()

    def _infer_pyomo_code(self, five_elem: str) -> str:
        """
        Generates Pyomo code from the extracted five elements.

        Args:
            five_elem: The string representing the five elements of the optimization problem.

        Returns:
            A string of Pyomo-compatible Python code.
        """
        messages = [{"role": "user", "content": generate_prompt.F2C(five_elem)}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=8192)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        code = response.replace("\\\\n", "\n").replace("&#39;", "'") \
                    .replace("&lt;", "<").replace("&gt;", ">") \
                    .replace("\\\\\"", "\"")

        return code.split("```python")[1].split("```")[0] \
                .replace('print("\\\\\n', 'print("') \
                .replace('print(f"\\\\\n', 'print(f"')

    def forward(self, problem_description: str) -> str:
        """
        Generates Pyomo code from a natural language optimization problem description.

        Args:
            problem_description: A natural language description of the optimization problem.

        Returns:
            A string containing the generated Pyomo code, with the five-element description as comments at the top.
        """
        five_elem = self._infer_five_elements(problem_description)
        pyomo_code = self._infer_pyomo_code(five_elem)
        # Add section comments for clarity
        five_elem_comment = "# === Five-element problem description ===\n" + \
                            "\n".join(f"# {line}" for line in five_elem.strip().splitlines())
        pyomo_code_comment = "# === Pyomo code ==="
        return f"{five_elem_comment}\n\n{pyomo_code_comment}\n{pyomo_code}"