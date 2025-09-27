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

"""Tests the persistence capabilities of the DatastoreSessionService.

This test verifies that a conversation history can be successfully persisted to
Google Cloud Datastore and resumed in a subsequent session.
"""

import asyncio
import os
import uuid

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.genai import types

from adk_datastore_session.datastore_session_service import (
    DataStoreKeys, DatastoreSessionService)

load_dotenv()

# --- Test Configuration ---
GCP_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
APP_NAME = "datastore-persistence-test"
AGENT_INSTRUCTION = "You are a helpful assistant. Your goal is to remember information given to you and recall it when asked. When asked for a secret code, you must repeat it back exactly as it was given to you."


async def test_datastore_persistence():
    """Runs a two-part test to verify conversation history is persisted in Datastore."""
    if not GCP_PROJECT_ID:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set.")

    session_id = str(uuid.uuid4())
    secret_code = f"CODE-{uuid.uuid4()!s}"

    # --- STEP 1: Create a session and provide a secret code. ---
    print("--- STEP 1: Storing the secret code in Datastore ---")
    try:
        # Instantiate a new runner and service pointing to Datastore.
        session_service_1 = DatastoreSessionService(project=GCP_PROJECT_ID, database="adktest")
        agent_1 = Agent(
            model="gemini-2.5-flash", name="TestAgent", instruction=AGENT_INSTRUCTION
        )
        runner_1 = Runner(
            agent=agent_1, app_name=APP_NAME, session_service=session_service_1
        )

        # Explicitly create the session in Datastore.
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
        session_service_2 = DatastoreSessionService(project=GCP_PROJECT_ID, database="adktest")
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

        print(
            "\nSUCCESS: Agent correctly recalled the secret code from the Datastore session."
        )

    except Exception as e:
        print(f"Error in Step 2: {e}")
        raise

    finally:
        # --- CLEANUP: Delete the test session from Datastore. ---
        print("\n--- Cleaning up test data from Datastore ---")
        from google.cloud import datastore

        ds_client = datastore.Client(project=GCP_PROJECT_ID)
        keys = DataStoreKeys(ds_client)
        session_key = keys.create_session_key(APP_NAME, session_id, session_id)

        # Deleting the session key also deletes descendant events.
        ds_client.delete(session_key)
        print(f"Deleted test session {session_id} from Datastore.")


if __name__ == "__main__":
    # This test requires an internet connection and authenticated gcloud credentials.
    asyncio.run(test_datastore_persistence())
