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

"""Tests the functional equivalence between DatabaseSessionService and DatastoreSessionService.

This test verifies that both services can be used to have a conversation with an agent
and that the agent can recall information from the session history, regardless of the
underlying storage mechanism.
"""

import asyncio
import os
import uuid

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from adk_datastore_session.datastore_session_service import DatastoreSessionService

load_dotenv()

# --- Test Configuration ---
GCP_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
DB_URL = "sqlite:///test_equivalence.db"
APP_NAME = "equivalence-test-app"
AGENT_INSTRUCTION = "You are a helpful assistant. Your primary goal is to remember all information given to you and recall it when asked. When asked for specific pieces of information like a name, a code, or JSON data, you must repeat it back exactly as it was given to you."


async def run_turn(runner: Runner, user_id: str, session_id: str, prompt: str) -> str:
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


async def test_functional_equivalence():
    """Runs a side-by-side validation of the two session services."""
    if not GCP_PROJECT_ID:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set.")

    user_id = f"test-user-{uuid.uuid4()!s}"
    session_id = str(uuid.uuid4())
    secret_code = f"SECRET_{uuid.uuid4()!s}"

    # 1. Instantiate both session services.
    db_service = DatabaseSessionService(db_url=DB_URL)
    ds_service = DatastoreSessionService(project=GCP_PROJECT_ID, database="adktest2")

    agent = Agent(
        model="gemini-2.5-flash", name="TestAgent", instruction=AGENT_INSTRUCTION
    )

    db_runner = Runner(agent=agent, app_name=APP_NAME, session_service=db_service)
    ds_runner = Runner(agent=agent, app_name=APP_NAME, session_service=ds_service)
    runners = {"Database": db_runner, "Datastore": ds_runner}

    try:
        # 2. Create sessions for both services.
        print("--- Creating sessions for both services ---")
        for runner in runners.values():
            await runner.session_service.create_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id
            )

        # 3. Run a conversation to provide the secret code.
        print("\n--- Storing secret code with both services ---")
        prompt = f"My secret code is {secret_code}."
        for name, runner in runners.items():
            print(f"\n-- Running for {name} --")
            await run_turn(runner, user_id, session_id, prompt)

        # 4. In a new "session", ask for the secret code and verify.
        print("\n--- Recalling secret code with both services ---")
        recall_prompt = "What is my secret code?"
        for name, runner in runners.items():
            print(f"\n-- Running for {name} --")
            # Use a new runner to ensure history is loaded from persistence
            new_runner = Runner(agent=agent, app_name=APP_NAME, session_service=runner.session_service)
            response = await run_turn(new_runner, user_id, session_id, recall_prompt)
            assert secret_code in response, f"Agent using {name} failed to recall the secret code."
            print(f"SUCCESS: Agent using {name} correctly recalled the secret code.")

        print("\nVALIDATION SUCCEEDED: Both services can persist and recall information.")

    finally:
        # 5. Clean up resources.
        print("\n--- Cleaning up resources ---")
        db_file = DB_URL.replace("sqlite:///", "")
        try:
            if os.path.exists(db_file):
                os.remove(db_file)
                print(f"Removed test database: {db_file}")
        except PermissionError:
            print(f"Warning: Could not remove test database {db_file} because it is in use.")

        from google.cloud import datastore

        ds_client = datastore.Client(project=GCP_PROJECT_ID, database="adktest2")
        user_key = ds_client.key("ADKStorageUserState", user_id, parent=ds_client.key("ADKStorageAppState", APP_NAME))
        ds_client.delete(user_key)
        print(f"Deleted datastore entities for user {user_id}")


if __name__ == "__main__":
    asyncio.run(test_functional_equivalence())
