# The MIT License (MIT)
#
# Copyright (c) 2025 pentium10
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tests the persistence capabilities of the FirestoreSessionService.

This test verifies that a conversation history can be successfully persisted to
Google Cloud Firestore and resumed in a subsequent session.
"""

import os
import uuid

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.genai import types

from adk_firestore_session.firestore_session_service import \
    FirestoreSessionService

load_dotenv()

# --- Test Configuration ---
GCP_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
APP_NAME = "firestore-persistence-test"
AGENT_INSTRUCTION = "You are a helpful assistant. Your primary goal is to remember all information given to you and recall it when asked. When asked for specific pieces of information like a name, a code, or JSON data, you must repeat it back exactly as it was given to you."


async def run_turn(
    runner: Runner, user_id: str, session_id: str, prompt: str
) -> str:
    """Helper function to run a single turn of conversation and return the agent's response."""
    print(f">>> User: {prompt}")
    user_message = types.Content(role="user", parts=[types.Part(text=prompt)])
    final_response = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=user_message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
    print(f"<<< Agent: {final_response}")
    return final_response


async def test_firestore_persistence():
    """Runs a two-part test to verify conversation history is persisted in Firestore."""
    if not GCP_PROJECT_ID:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set.")

    user_id = f"test-user-{uuid.uuid4()!s}"
    session_id = str(uuid.uuid4())
    secret_code = f"SECRET_{uuid.uuid4()!s}"
    fs_service = None

    agent = Agent(
        model="gemini-2.5-flash", name="TestAgent", instruction=AGENT_INSTRUCTION
    )

    try:
        # --- STEP 1: Create a session and provide a secret code. ---
        print("--- STEP 1: Storing the secret code in Firestore ---")
        fs_service = FirestoreSessionService(project=GCP_PROJECT_ID)
        runner_1 = Runner(agent=agent, app_name=APP_NAME, session_service=fs_service)

        # Create the session.
        await runner_1.session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )

        # Run a conversation to provide the secret code.
        prompt = f"My secret code is {secret_code}."
        await run_turn(runner_1, user_id, session_id, prompt)
        print("Step 1 complete. Agent has processed the secret.")

        # --- STEP 2: Resume the session and ask for the secret code. ---
        print("\n--- STEP 2: Resuming session and recalling the secret ---")
        # Use a new runner to ensure history is loaded from persistence
        fs_service_2 = FirestoreSessionService(project=GCP_PROJECT_ID)
        runner_2 = Runner(
            agent=agent, app_name=APP_NAME, session_service=fs_service_2
        )

        recall_prompt = "What is my secret code?"
        response = await run_turn(runner_2, user_id, session_id, recall_prompt)

        assert (
            secret_code in response
        ), f"Agent failed to recall the secret code."

        print(
            "\nSUCCESS: Agent correctly recalled the secret code from the Firestore session."
        )

    finally:
        # --- CLEANUP: Delete the test session from Firestore. ---
        print("\n--- Cleaning up resources ---")
        if fs_service:
            await fs_service.delete_session(APP_NAME, user_id, session_id)
            print(f"Deleted firestore entities for session {session_id}")
