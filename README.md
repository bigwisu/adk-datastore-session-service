# ADK Firestore Session Service

A Python library that provides a community extension to the ADK (Agent Development Kit) for session persistence using Google Cloud Firestore (in Datastore Mode).

## Key Features

*   **Firestore Integration:** Uses Google Cloud Firestore (in Datastore Mode) for persistent session storage.
*   **Seamless ADK Extension:** Integrates directly with the ADK `BaseSessionService`.
*   **Storage Optimization:** Automatically compresses event content and can strip large binary data to stay within the 1MB entity/document size limit.

## Installation

```bash
pip install adk-cloud-session-service
```

Alternative pip:
```bash
pip install git+https://github.com/pentium10/adk-datastore-session-service.git
```

Using uv:
```bash
uv pip install "git+https://github.com/pentium10/adk-datastore-session-service"
```

## Configuration

The ADK framework and this library rely on environment variables for configuration. While you can pass settings directly to the constructor, the recommended approach is to place them in a `.env` file in your project's root directory. The tests in this repository use this method.

```
# .env file
GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
GOOGLE_API_KEY="your-google-api-key"
```

The `DatastoreSessionService` can be customized through its constructor:

```python
from adk_datastore_session import DatastoreSessionService, DatastoreKeyNames

project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")

session_service = DatastoreSessionService(
    project=project_id,                 # Required: Your Google Cloud project ID.
    database="(default)",               # Optional: The Datastore database ID. Defaults to "(default)".
    key_names=DatastoreKeyNames(...),   # Optional: Customize the entity kind names used in Datastore.
    strip_large_content=True            # Optional: If True, replaces large inline data (like images) with a 1x1 transparent PNG placeholder to prevent errors and avoid disrupting web UIs.
)
```

### Storage and Size Optimization

To optimize storage and stay within Google Cloud Datastore's limits, this library includes two key features:

1.  **Gzip Compression:** The content of each event is automatically compressed using `gzip` before being stored. This process is handled transparently by the service.
2.  **Content Stripping:** When `strip_large_content` is enabled (the default), the service replaces the data of any large, inline content (like images from a multimodal prompt) with a tiny, 1x1 transparent PNG. This prevents errors from exceeding Datastore's size limits while ensuring that frontend UIs do not break when rendering the event history.

### Customizing Entity Kinds

You can change the names (Kinds) of the entities stored in Datastore by passing a `DatastoreKeyNames` object. The default kinds are `ADKStorageAppState`, `ADKStorageUserState`, `ADKStorageSession`, and `ADKStorageEvent`.

```python
from adk_datastore_session import DatastoreSessionService, DatastoreKeyNames

# Define your custom entity kind names
custom_key_names = DatastoreKeyNames(
    app_prefix="ADKStorageAppState",
    user_prefix="ADKStorageAppState",
    session_prefix="ADKStorageSession",
    event_prefix="ADKStorageEvent"
)

# Instantiate the service with your custom key names
session_service = DatastoreSessionService(
    project="your-gcp-project-id",
    key_names=custom_key_names
)
```

## Usage

The following example demonstrates how to create a session, store information, and have the agent recall it from persistence in a subsequent run.

```python
# samples/example.py

import asyncio
import os
import uuid

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.genai import types

from adk_datastore_session.datastore_session_service import DatastoreSessionService


async def main():
    """Demonstrates creating a session, persisting data, and resuming it."""

    # --- Configuration ---
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
        session_service_1 = DatastoreSessionService(project=project_id)
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
        session_service_2 = DatastoreSessionService(project=project_id)
        runner_2 = Runner(agent=agent, app_name=app_name, session_service=session_service_2)

        print("User says: 'what is the secret code?'")
        user_message_2 = types.Content(role='user', parts=[types.Part(text="what is the secret code?")])
        final_response_text = ""

        async for event in runner_2.run_async(user_id=user_id, session_id=session_id, new_message=user_message_2):
            if event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
                break
        
        print(f"Agent responded: '{final_response_text}'")
        
        assert secret_code in final_response_text, f"Agent response did not contain the secret code '{secret_code}'"
        
        print("\nSUCCESS: Agent correctly recalled the secret code from the Datastore session.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # --- CLEANUP ---
        print(f"\nCleaning up by deleting session: {session_id}")
        await session_service_1.delete_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        print("Cleanup complete.")


if __name__ == "__main__":
    # Ensure you have a .env file with GOOGLE_CLOUD_PROJECT and GOOGLE_API_KEY
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())

```

## Datastore Indexes

This library automatically creates the necessary Datastore entity kinds when it runs. To query the event history efficiently, a composite index is required. The service will attempt to create this index for you automatically on its first query.

However, this automatic creation can fail if the service account or user credentials do not have the `datastore.indexes.create` permission.

If the automatic creation fails, you must create the index manually. The required index definition is included in the source code at `src/index.yaml`. **Note:** If you customize the entity kinds using `DatastoreKeyNames`, you must update the `kind` in this file to match your `event_prefix`.

```yaml
# src/index.yaml
indexes:
- kind: ADKStorageEvent
  ancestor: yes
  properties:
  - name: timestamp
    direction: desc
```

You can create this index using the `gcloud` CLI:

```bash
gcloud datastore indexes create src/index.yaml
```

## Contributing

Interested in contributing to the development of this library? Follow these steps to set up your development environment.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install dependencies in editable mode:**
    ```bash
    pip install -e ".[dev]"
    ```

### Running Tests

This project uses `pytest` for testing. Before running the tests, you must create a `.env` file in the project root with your Google Cloud credentials (see `.env.example`).

Once your `.env` file is configured, run the test suite with:

```bash
pytest
```
