import json
from mlx_lm import load, generate

# 1. Load the local 8-bit quantized MLX model and tokenizer
# You can also use the 16bit by choosing this path : "hskyto/lfm2.5-2.6b-toolcall-mlx" 
model_path = "Hskyto/lfm2.5-2.6b-toolcall-mlx-q8" 

print(f"Loading local model from {model_path}...")
model, tokenizer = load(model_path)

# 2. Define the on-device iOS Tool Registry (OpenAI/xLAM compatible schema)
IOS_TOOLS = [
    {
        "name": "ios.create_reminder",
        "description": "Create a new Apple Reminder with text and time.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The content or title of the reminder.",
                },
                "time": {
                    "type": "string",
                    "description": "Time normalized to 24-hour format e.g. '18:00'.",
                },
            },
            "required": ["text", "time"],
        },
    },
    {
        "name": "ios.schedule_event",
        "description": "Schedule a calendar event with a title, start time, and duration.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title of the calendar event.",
                },
                "start_time": {
                    "type": "string",
                    "description": "Start time in ISO or 24-hour format.",
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Duration of the event in minutes.",
                },
            },
            "required": ["title", "start_time"],
        },
    },
    {
        "name": "ios.send_message",
        "description": "Send an iMessage or text message to a contact.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Name of the contact."},
                "message": {
                    "type": "string",
                    "description": "The body text of the message.",
                },
            },
            "required": ["recipient", "message"],
        },
    },
    {
        "name": "ios.set_alarm",
        "description": "Set a device alarm for a specific time with an optional label.",
        "parameters": {
            "type": "object",
            "properties": {
                "time": {
                    "type": "string",
                    "description": "Alarm time in 24-hour format, e.g. '07:30'.",
                },
                "label": {"type": "string", "description": "Label for the alarm."},
            },
            "required": ["time"],
        },
    },
    {
        "name": "ios.adjust_system_setting",
        "description": "Modify device control center settings like brightness, volume, or connectivity.",
        "parameters": {
            "type": "object",
            "properties": {
                "setting": {
                    "type": "string",
                    "enum": [
                        "brightness",
                        "volume",
                        "wifi",
                        "bluetooth",
                        "flashlight",
                        "low_power_mode",
                    ],
                    "description": "The setting to change.",
                },
                "value": {
                    "type": "string",
                    "description": "The target state or level (e.g., 'on', 'off', '50%').",
                },
            },
            "required": ["setting", "value"],
        },
    },
    {
        "name": "ios.start_timer",
        "description": "Start a countdown timer for a specific duration.",
        "parameters": {
            "type": "object",
            "properties": {
                "duration_seconds": {
                    "type": "integer",
                    "description": "Duration of the timer in seconds.",
                },
                "label": {
                    "type": "string",
                    "description": "Optional label for the timer.",
                },
            },
            "required": ["duration_seconds"],
        },
    },
]

# 3. 12 Real-World On-Device Test Cases
TEST_CASES = [
    "Remind me to call Mom at 6pm",
    "Wake me up tomorrow at 7:30am labeled Workout",
    "Turn on the flashlight and set screen brightness to 50%",
    "Schedule a meeting called 'Project Sync' for tomorrow at 14:00 for 45 minutes",
    "Text Sarah that I will be running 10 minutes late",
    "Set a countdown timer for 15 minutes for boiling pasta",
    "Turn off Wi-Fi and enable low power mode",
    "Add a reminder to pick up groceries at 5:30pm",
    "Schedule a dentist appointment on Friday at 10am for 60 minutes",
    "Text John happy birthday!",
    "Set an alarm for 06:00 called Early Flight",
    "Turn off Bluetooth",
]


# 4. Run through the cases using the model's native chat template
def run_evaluation():
    print(f"\n--- Running {len(TEST_CASES)} On-Device iOS Test Cases ---")

    for i, query in enumerate(TEST_CASES, 1):
        print(f'\n[Test Case {i}/{len(TEST_CASES)}] Query: "{query}"')

        # Build chat messages using the model's chat template structure with tools
        messages = [
            {
                "role": "system",
                "content": "You are an expert on-device AI assistant with access to local iOS tools. Select the correct tool(s) and provide clean arguments.",
            },
            {"role": "user", "content": query},
        ]

        # Apply the model's tokenizer chat template, injecting the tool definitions
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tools=IOS_TOOLS, tokenize=False, add_generation_prompt=True
            )
        except TypeError:
            # Fallback if tokenizer signature handles tools differently in specific transformers versions
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Generate response using MLX
        response = generate(
            model, tokenizer, prompt=prompt, max_tokens=256, verbose=False
        )

        print(f"-> Model Output:\n{response.strip()}")
        print("-" * 50)


if __name__ == "__main__":
    run_evaluation()
