
import os
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Union
import json

# Load the .env file
load_dotenv()

# Set the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

class Instruction(BaseModel):
    text: str = Field(..., description="Generated instruction.")
    pause: str = Field(..., description="Pause time in milliseconds.")

class Instructions(BaseModel):
    instructions: List[Instruction]

    model_config = {
        "strict": True  # This enables strict mode
    }

async def generate_instructions(prompt: str, model: str = "gpt-4-0613") -> Union[Instructions, dict]:
    """Generates an array of meditation instructions with pause times."""
    
    # Define the tool for structured output
    tools = [{
        "type": "function",
        "function": {
            "name": "generate_instruction_with_pause",
            "description": "Generates an instruction with an associated pause time.",
            "parameters": Instructions.model_json_schema()
        }
    }]
    
    try:
        # The OpenAI API call with the tool
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an assistant that generates step-by-step meditation instructions with pauses. The instruction will be focused on the type of meditation provided."},
                {"role": "user", "content": prompt}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "generate_instruction_with_pause"}},
        )
        response_message = response['choices'][0]['message']
        
        if 'tool_calls' in response_message:
            tool_call = response_message['tool_calls'][0]
            if tool_call['function']['name'] == 'generate_instruction_with_pause':
                # return Instructions.model_validate_json(tool_call['function']['arguments'])  
                instructions_dict = json.loads(tool_call['function']['arguments'])
                return instructions_dict
        return {"error": "No valid tool call made."}
    
    except openai.error.OpenAIError as e:
        return {"error": f"OpenAI API error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

class AllMeditationPrompts:
    """Prompt for meditation types."""

    mindful_meditation_prompt = """Give me a step-by-step guide on how to do {focus_area} type meditation. The duration of the meditation will be {duration} milliseconds. Add pauses after each instruction as you see fit. Make sure to give simple and easy to understand instructions. Do not include jargon."""

