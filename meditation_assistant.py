import os
import openai
from dotenv import load_dotenv
# Load the .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


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
            "required": ["text", "pause"],
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
    print(response)
    # Output is parsed
    return response.choices[0].message['function_call']['arguments']





class AllMeditationPrompts:
    mindful_meditation_prompt = """Give me a step-by-step guide on how to do {focus_area} type meditation. The duration of the medation will be {duration} milliseconds. Add pauses after each instructions as you see fit. Make sure to give simple and easy to understand instructions. Do not include jargon."""