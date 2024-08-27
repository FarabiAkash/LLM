import unittest
from unittest.mock import patch, MagicMock
import json
import asyncio
from meditation_assistant import generate_instructions, AllMeditationPrompts

class TestMeditationAssistant(unittest.IsolatedAsyncioTestCase):

    # Test for successful instructions
    @patch('meditation_assistant.openai.ChatCompletion.create')
    async def test_generate_instructions_success(self, mock_create):
        mock_response = {
            'choices': [{
                'message': {
                    'function_call': {
                        'arguments': json.dumps({
                            "instructions": [
                                {"text": "Sit comfortably", "pause": "5000"},
                                {"text": "Take a deep breath", "pause": "3000"}
                            ]
                        })
                    }
                }
            }]
        }
        mock_create.return_value = mock_response

        prompt = "Test prompt"
        result = await generate_instructions(prompt)
        
        self.assertIn("instructions", result)
        instructions = json.loads(result)["instructions"]
        self.assertEqual(len(instructions), 2)
        self.assertEqual(instructions[0]["text"], "Sit comfortably")
        self.assertEqual(instructions[0]["pause"], "5000")

    # Test for request refusal
    @patch('meditation_assistant.openai.ChatCompletion.create')
    async def test_generate_instructions_refusal(self, mock_create):
        mock_response = {
            'choices': [{
                'message': {
                    'refusal': "Unable to generate instructions"
                }
            }]
        }
        mock_create.return_value = mock_response

        prompt = "Test prompt"
        result = await generate_instructions(prompt)
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Unable to generate instructions")

    # Test for unexpected response
    @patch('meditation_assistant.openai.ChatCompletion.create')
    async def test_generate_instructions_unexpected_response(self, mock_create):
        mock_response = {
            'choices': [{
                'message': {
                    'content': "Unexpected response"
                }
            }]
        }
        mock_create.return_value = mock_response

        prompt = "Test prompt"
        result = await generate_instructions(prompt)
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "No function call made.")

    # Test for prompt formatting
    def test_mindful_meditation_prompt(self):
        duration = 45000
        focus_area = 'relaxation'
        expected_prompt = f"Give me a step-by-step guide on how to do relaxation type meditation. The duration of the medation will be 45000 milliseconds. Add pauses after each instructions as you see fit. Make sure to give simple and easy to understand instructions. Do not include jargon."
        
        formatted_prompt = AllMeditationPrompts.mindful_meditation_prompt.format(
            duration=duration, 
            focus_area=focus_area
        )
        
        self.assertEqual(formatted_prompt, expected_prompt)

    # Test for invalid input (Bad path)
    @patch('meditation_assistant.openai.ChatCompletion.create')
    async def test_generate_instructions_invalid_input(self, mock_create):
        mock_create.side_effect = ValueError("Invalid input")

        prompt = "Invalid prompt"
        with self.assertRaises(ValueError):
            await generate_instructions(prompt)

if __name__ == '__main__':
    unittest.main()