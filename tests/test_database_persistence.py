# The MIT License (MIT)
#
# Copyright (c) 2025 pentium10
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tests the persistence capabilities of the DatabaseSessionService.

This script serves as a baseline to ensure that the standard ADK
DatabaseSessionService correctly persists and resumes conversation history.
"""

import asyncio
import os
import uuid

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

load_dotenv()

# --- Test Configuration ---
DB_URL = "sqlite:///test_persistence.db"
APP_NAME = "persistence-test-app"
AGENT_INSTRUCTION = "You are a helpful assistant. Your goal is to remember information given to you and recall it when asked. When asked for a secret code, you must repeat it back exactly as it was given to you."


async def test_database_persistence():
    """Runs a two-part test to verify conversation history is persisted in a database."""
    session_id = str(uuid.uuid4())
    secret_code = f"CODE-{uuid.uuid4()!s}"

    # --- STEP 1: Create a session and provide a secret code. ---
    print("--- STEP 1: Storing the secret code in the database ---")
    try:
        # Instantiate a new runner and service.
        session_service_1 = DatabaseSessionService(db_url=DB_URL)
        agent_1 = Agent(
            model="gemini-2.5-flash", name="TestAgent", instruction=AGENT_INSTRUCTION
        )
        runner_1 = Runner(
            agent=agent_1, app_name=APP_NAME, session_service=session_service_1
        )

        # Explicitly create the session.
        print(f"Creating new session: {session_id}")
        await session_service_1.create_session(
            app_name=APP_NAME, user_id=session_id, session_id=session_id
        )

        # Send the secret code to the agent.
        print(f"User says: 'the secret code is {secret_code}'")
        user_message_1 = types.Content(
            role="user", parts=[types.Part(text=f"the secret code is {secret_code}")]
        )
        async for _ in runner_1.run_async(
            user_id=session_id, session_id=session_id, new_message=user_message_1
        ):
            pass
        print("Step 1 complete. Agent has processed the secret.")

    except Exception as e:
        print(f"Error in Step 1: {e}")
        raise

    # --- STEP 2: Resume the session and ask for the secret code. ---
    print("\n--- STEP 2: Resuming session and recalling the secret ---")
    try:
        # Instantiate a new runner and service to simulate resuming the conversation.
        session_service_2 = DatabaseSessionService(db_url=DB_URL)
        agent_2 = Agent(
            model="gemini-2.5-flash", name="TestAgent", instruction=AGENT_INSTRUCTION
        )
        runner_2 = Runner(
            agent=agent_2, app_name=APP_NAME, session_service=session_service_2
        )

        # Ask the agent for the secret code.
        print("User says: 'what is the secret code?'")
        user_message_2 = types.Content(
            role="user", parts=[types.Part(text="what is the secret code?")]
        )
        final_response_text = ""

        async for event in runner_2.run_async(
            user_id=session_id, session_id=session_id, new_message=user_message_2
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
                break

        print(f"Agent responded: '{final_response_text}'")

        # Verify that the agent recalled the secret code correctly.
        assert (
            secret_code in final_response_text
        ), f"Agent response did not contain the secret code '{secret_code}'"

        print("\nSUCCESS: Agent correctly recalled the secret code from the database session.")

    except Exception as e:
        print(f"Error in Step 2: {e}")
        raise

    finally:
        # --- CLEANUP: Remove the test database file. ---
        print("\n--- Cleaning up test data ---")
        db_file = DB_URL.replace("sqlite:///", "")
        try:
            if os.path.exists(db_file):
                os.remove(db_file)
                print(f"Removed test database: {db_file}")
        except PermissionError:
            print(f"Warning: Could not remove test database {db_file} because it is in use.")


if __name__ == "__main__":
    # This test requires an internet connection and a configured Gemini API key.
    asyncio.run(test_database_persistence())
