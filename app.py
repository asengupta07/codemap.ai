import chainlit as cl
from helpers import generate

@cl.on_message
async def main(message: cl.Message):    
    history = cl.chat_context.to_openai()[-10:]
    formatted_history = ""
    for entry in history:
        role = entry["role"]
        content = entry["content"]
        formatted_history += f"{role}: {content}\n"

    history_text = formatted_history

    response = await generate(message.content, history_text)

    await cl.Message(
        content=f"{response['result']}",
    ).send()

@cl.on_stop
async def on_stop():
    print("The user wants to stop the task!")
    await cl.Message(
        content="Generation stopped!",
    ).send()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="ML Roadmap",
            message="How can I become a machine learning engineer?",
            icon="/public/ml.svg",
            ),

        cl.Starter(
            label="WebDev Roadmap",
            message="What's the best way to start learning web development?",
            icon="/public/webdev.svg",
            ),
        cl.Starter(
            label="Blockchain Roadmap",
            message="How can I start a career in blockchain?",
            icon="/public/blockchain.svg",
            ),
        cl.Starter(
            label="Cybersecurity Roadmap",
            message="What's the best way to start a career in cybersecurity?",
            icon="/public/cybersec.svg",
            )
        ]