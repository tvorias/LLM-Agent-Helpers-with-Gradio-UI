# LLM-Agent-Helpers-with-Gradio-UI
Conversation-Aware LLM Utilities for Agent Development &amp; Learning with Gradio UI

# LLM Agent Helpers
**Conversation-Aware LLM Utilities for Agent Development & Learning**

A lightweight Python toolkit for interacting with large language models in a stateful, notebook-friendly way, designed to support hands-on development of LLM agents and workflows.

This project demonstrates practical use of the OpenAI Python SDK, conversational memory patterns, and local development erogonomics for AI engineering work. 

## Overview

**LLM Agent Helpers** provides a simple but extensible interface for asking technical questions of an LLM while maintaining conversational context across calls. It is designed for:
  - Interactive development in Jupyter notebooks (VS Code compatible)
  - Learning and experimenting with agent-like interaction patterns
  - Building intuition around LLM clients, state, and system prompting
  - Supportive iterative AI engineering workflows

The core abstraction is intentionally minimal: a conversation-aware helper function that behaves like the backbone of an AI agent without unnecessary complexity. 

## Motivation

This project originated from Week 2 of **Ed Donner's *AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents* ** course, which asked learners to:
  Build a tool that takes a technical question and responds with an explanation using OpenAI and Ollama. Include a Gradio UI. 

This code builds off the code in my 'LLM-Agent-Helpers' repo by adding the ability to display an interactive chatbox where a user can see the full history of the conversation. 

Rather than building a one-off script, I designed a reusable helper that:
  - Maintains conversational memory
  - Can be customized with domain-specific system prompts
  - Works with both hosted and local LLMs
  - Is structured like a real engineering utility, not a demo snippet

The result is a practical development tool I actively use while completing course exercises and prototyping agentic systems.

## What this project demonstrates
- Correct use of the OpenAI Python SDK
- Conversation state management without external storage
- System prompt design for scoped, task-focused responses
- Clean separation between public API and internal state
- Import-safe module design for notebooks and scripts
- Local development patterns for AI tooling

This repository intentionally avoids:
- Raw HTTPS calls (requests)
- Hardcoded API keys
- Framework-heavy abstractions
- Over-engineered agent layers

## Core Functionality
`ask_question(question: str, max_messages: int=20)`
  - Sends a user question to the LLM
  - Includes prior conversation turns for context
  - Trims history to prevent unbounded growth
  - Returns a Markdown-rendered response for notebooks or an interactive chatbox that takes in user input and allows a user to see the entire chat conversation
This function behaves like a **single-turn interface backed by multi-turn memory**, a foundational pattern in agent design.

`reset_memory()`
  - Clears conversation history

## Example Usage

Review the `test.ipynb` notebook for example usage. Responses are displayed directly in the notebook with preserved context.

## Environment Setup
Create a `.env` file in the project root. Review the .env.example file for the structure.