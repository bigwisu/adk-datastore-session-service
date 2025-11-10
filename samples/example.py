# Copyright 2025 pentium10
#
# The MIT License (MIT)
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

import asyncio
import os
import uuid

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.genai import types
from dotenv import load_dotenv
from adk_firestore_session.firestore_session_service import \
    FirestoreSessionService


async def main():
    """Demonstrates creating a session, persisting data, and resuming it."""

    # --- Configuration ---
    # The ADK and this session service rely on environment variables for configuration.
    # Ensure you have a .env file or have set the variables in your environment.
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("The GOOGLE_CLOUD_PROJECT environment variable is not set.")

    app_name = "my-adk-app"
    user_id = f"user-{uuid.uuid4()!s}"
    session_id = f"session-{uuid.uuid4()!s}"
    secret_code = f"SECRET-{uuid.uuid4()!s}"

    agent = Agent(
        model="gemini-2.5-flash",
        name="TestAgent",
        instruction="You are a helpful assistant. Your goal is to remember information given to you and recall it when asked. When asked for a secret code, you must repeat it back exactly as it was given to you."
    )

    try:
        # --- STEP 1: Create a session and tell the agent the secret code ---
        print(f"--- STEP 1: Storing a secret code in session {session_id} ---")
        session_service_1 = FirestoreSessionService(project=project_id)
        runner_1 = Runner(agent=agent, app_name=app_name, session_service=session_service_1)

        await runner_1.session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )

        print(f"User says: 'the secret code is {secret_code}'")
        user_message_1 = types.Content(role='user', parts=[types.Part(text=f"the secret code is {secret_code}")])
        async for _ in runner_1.run_async(user_id=user_id, session_id=session_id, new_message=user_message_1):
            pass
        print("Step 1 complete. Agent has processed the secret.")

        # --- STEP 2: Resume the session and ask for the secret code ---
        print("\n--- STEP 2: Resuming session and recalling the secret ---")
        # Instantiate a new runner and service to simulate resuming the conversation.
        session_service_2 = FirestoreSessionService(project=project_id)
        runner_2 = Runner(agent=agent, app_name=app_name, session_service=session_service_2)

        print("User says: 'what is the secret code?'")
        user_message_2 = types.Content(role='user', parts=[types.Part(text="what is the secret code?")])
        final_response_text = ""

        async for event in runner_2.run_async(user_id=user_id, session_id=session_id, new_message=user_message_2):
            if event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
                break
        
        print(f"Agent responded: '{final_response_text}'")
        
        assert secret_code in final_response_text, (
            f"Agent response did not contain the secret code '{secret_code}'"
        )
        
        print("\nSUCCESS: Agent correctly recalled the secret code from the Firestore session.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # --- CLEANUP ---
        if 'runner_1' in locals():
            print(f"\nCleaning up by deleting session: {session_id}")
            await runner_1.session_service.delete_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
            print("Cleanup complete.")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())