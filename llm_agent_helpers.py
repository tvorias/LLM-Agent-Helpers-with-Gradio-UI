from openai import OpenAI
from IPython.display import Markdown, display
import os
from dotenv import load_dotenv
import gradio as gr


def ask_question(
        question,
        conversation_history=None,
        ui_mode=True,
        max_messages=20
):
    """
    AI chatbot inference function with optional UI rendering.

    Parameters
    ----------
    question : str
        User input query.

    conversation_history : list or None
        Existing conversation history. If None, a new history is created.

    ui_mode : bool
        If True, function returns outputs suitable for Gradio UI.
        If False, function displays result using notebook display().

    max_messages : int
        Maximum number of conversation messages to retain.

    Returns
    -------
    If ui_mode=False:
        dict with 'text' and 'history'.

    If ui_mode=True:
        launches Gradio interface inside the notebook.
    """


    # Load environment variables once
    load_dotenv(override=True)

    # Initialize client
    if not hasattr(ask_question, "client"):
        ask_question.client = OpenAI()
    
    if not hasattr(ask_question, "conversation_history"):
        system_prompt = (
            """You are a Python and data science assistant specializing in large language models (LLMs)
            and AI agentic workflows.

            You ONLY provide answers related to:
            - Python software development
            - Data science workflows
            - LLM APIs and implementations
            - AI agent design and orchestration

            ALL LLM interactions MUST use the OpenAI Python SDK.
            The word "client" ALWAYS refers to a software API client.
            """
        )

        ask_question.conversation_history = [
            {"role": "system", "content": system_prompt}
        ]

    if not ui_mode:
        if question is None:
            raise ValueError("Question must be provided when ui_mode=False")


        # Trim history if needed (keep system prompt)
        while len(ask_question.conversation_history) > max_messages:
            conversation_history.pop(1)

        # Add user question
        ask_question.conversation_history.append(
            {"role": "user", "content": question}
        )

        # Call the model
        response = ask_question.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=conversation_history
        )

        answer = response.choices[0].message.content

        # Store assistant reply
        ask_question.conversation_history.append(
            {"role": "assistant", "content": answer}
        )

        display(Markdown(answer))
        return {"text": answer, "history": ask_question.conversation_history}
    
    else:
        # Define Gradio callback
        def chat_ui(user_message, history):
            # Trim history
            while len(ask_question.conversation_history) > max_messages:
                ask_question.conversation_history.pop(1)

            # Add user message
            ask_question.conversation_history.append(
                {"role": "user", "content": user_message}
            )

            # Call LLM
            response = ask_question.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=ask_question.conversation_history
            )
            answer = response.choices[0].message.content

            # Store assistant reply
            ask_question.conversation_history.append(
                {"role": "assistant", "content": answer}
            )

            history.append((user_message, answer))
            return history, ""

        # Build Gradio UI
        with gr.Blocks() as ui:
            chatbot = gr.Chatbot()
            textbox = gr.Textbox(
                label="Chat with AI Assistant",
                placeholder="Type your question here..."
            )
            textbox.submit(
                chat_ui,
                inputs=[textbox, chatbot],
                outputs=[chatbot, textbox]
            )

        ui.launch(inbrowser=False, share=False)

def reset_memory():
    """Clear conversation history for ask_question."""
    if hasattr(ask_question, "conversation_history"):
        del ask_question.conversation_history
        del ask_question.client
        print("Conversation history reset.")
    else:
        print("No conversation history exists.")