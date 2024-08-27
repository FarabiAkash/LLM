import os
import openai
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Set the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


"""Generates an array of meditation instructions with pause times."""
async def generate_instructions(prompt, model="gpt-4"):
    # Function for the specific output
    function = [{
        "name": "generate_instruction_with_pause",
        "description": "Generates an instruction with an associated pause time.",
        "parameters": {
            "type": "object",
            "properties": {
                "instructions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Generated instruction."
                            },
                            "pause": {
                                "type": "string",
                                "description": "Pause time in milliseconds."
                            }
                        },
                        "required": ["text", "pause"]
                    }
                }
            },
            "required": ["instructions"],
            "strict": True
        }
    }]
    
    # The OpenAI API call with the function
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant that generates step-by-step medidation instructions with pauses. The instruction will be focused on the type of medidation provided."},
            {"role": "user", "content": prompt}
        ],
        functions=function,
        function_call="auto",
    )
    
    # Handle the response, including any refusal
    response_message = response['choices'][0]['message']
    
    # Check if there is a refusal key and if it is not null
    # print(response_message.get('refusal'))
    if response_message.get('refusal'):
        # print(f"Assistant's refusal: {response_message['refusal']}")
        return {"error": response_message['refusal']}
    else:
        # Parse the output if the function call is successful
        function_call = response_message.get('function_call')
        
        if function_call:
            # Return the function call arguments
            return function_call['arguments']
        else:
            return {"error": "No function call made."}




class AllMeditationPrompts:
    """Prompt for meditation types."""

    mindful_meditation_prompt = """Give me a step-by-step guide on how to do {focus_area} type meditation. The duration of the medation will be {duration} milliseconds. Add pauses after each instructions as you see fit. Make sure to give simple and easy to understand instructions. Do not include jargon."""