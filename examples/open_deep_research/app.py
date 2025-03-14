from .run import create_agent

from src.gradio_ui import GradioUI


agent = create_agent(model_id='gpt-4o-mini')

demo = GradioUI(agent)

if __name__ == "__main__":
    demo.launch()
