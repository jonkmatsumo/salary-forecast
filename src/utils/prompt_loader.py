import os

def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the 'prompts' directory. Args: prompt_name (str): Filename without extension. Returns: str: Prompt file content. Raises: FileNotFoundError: If prompt file not found."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_dir = os.path.join(current_dir, "..", "llm", "prompts")
    prompt_path = os.path.join(prompt_dir, f"{prompt_name}.md")
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()
