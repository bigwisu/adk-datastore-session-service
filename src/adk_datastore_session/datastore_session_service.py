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
from __future__ import annotations

import asyncio
import copy
import gzip
import json
import uuid
from datetime import datetime
from typing import Any, Optional

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import _session_util
from google.adk.sessions.base_session_service import (BaseSessionService,
                                                      GetSessionConfig,
                                                      ListSessionsResponse)
from google.adk.sessions.session import Session
from google.adk.sessions.state import State
from google.api_core.exceptions import FailedPrecondition
from google.cloud import datastore


class DatastoreKeyNames:
    """
    Example Usage:
    ```
    from datastore_session_service import DatastoreSessionService, DatastoreKeyNames

    class MyCustomKeyNames(DatastoreKeyNames):
        def __init__(self):
            super().__init__(
                app_prefix="ADKStorageAppState",
                user_prefix="ADKStorageUserState",
                session_prefix="ADKStorageSession",
                event_prefix="ADKStorageEvent"
            )

    # Instantiate the service with your custom key names
    custom_key_names = MyCustomKeyNames()
    session_service = DatastoreSessionService(
        project="your-gcp-project-id",
        database="your-datastore-database-id",
        key_names=custom_key_names
    )
    ```
    """

    def __init__(
        self,
        app_prefix="ADKStorageAppState",
        user_prefix="ADKStorageUserState",
        session_prefix="ADKStorageSession",
        event_prefix="ADKStorageEvent",
    ):
        self.app_prefix = app_prefix
        self.user_prefix = user_prefix
        self.session_prefix = session_prefix
        self.event_prefix = event_prefix


class DataStoreKeys:
    """Functions for creating Datastore keys."""

    def __init__(self, client, key_names: Optional[DatastoreKeyNames] = None):
        self.client = client
        if key_names is None:
            key_names = DatastoreKeyNames()
        self.app_prefix = key_names.app_prefix
        self.user_prefix = key_names.user_prefix
        self.session_prefix = key_names.session_prefix
        self.event_prefix = key_names.event_prefix

    def create_app_key(self, app_name: str):
        """Creates a Datastore key for a StorageAppState."""
        return self.client.key(self.app_prefix, app_name)

    def create_user_key(self, app_name: str, user_id: str):
        """Creates a Datastore key for a StorageUserState."""
        app_key = self.create_app_key(app_name)
        return self.client.key(self.user_prefix, user_id, parent=app_key)

    def create_session_key(
        self, app_name: str, user_id: str, session_id: Optional[str] = None
    ):
        """Creates a Datastore key for a StorageSession."""
        user_key = self.create_user_key(app_name, user_id)
        if session_id is None:
            session_id = str(uuid.uuid4())
        return self.client.key(self.session_prefix, session_id, parent=user_key)

    def create_event_key(
        self, app_name: str, user_id: str, session_id: str, event_id: str
    ):
        """Creates a Datastore key for a StorageEvent."""
        session_key = self.create_session_key(app_name, user_id, session_id)
        return self.client.key(self.event_prefix, event_id, parent=session_key)


class DatastoreSessionService(BaseSessionService):
    """A session service that uses Google Cloud Datastore for storage.

    This service stores ADK session data in Google Cloud Datastore. It provides
    several customization options and automatically compresses event content to
    optimize storage.

    Args:
        project: The Google Cloud project ID.
        database: The Datastore database ID. If not provided, the default
            database is used.
        key_names: An object to customize the entity kind names used in Datastore.
        strip_large_content: If True, replaces large inline data (like images)
            with a 1x1 transparent PNG placeholder to prevent errors and avoid
            disrupting web UIs.
    """

    def __init__(
        self,
        project: str,
        database: Optional[str] = None,
        key_names: Optional[DatastoreKeyNames] = None,
        strip_large_content: bool = True,
        **kwargs: Any,
    ):
        """Initializes the Datastore session service."""
        self.client = datastore.Client(project=project, database=database, **kwargs)
        self.key_factory = DataStoreKeys(self.client, key_names)
        self.strip_large_content = strip_large_content

    async def _handle_missing_index(self, kind_name: str):
        """Lazy-loads dependencies, triggers index creation, and waits for it to be ready."""
        # To reduce cold start time, dependencies are imported and classes are defined only when needed.
        import asyncio
        import time

        import google.auth
        import google.auth.transport.requests
        import requests

        class DatastoreIndexManager:
            """Manages the creation of Datastore indexes using the raw REST API."""

            def __init__(self, project_id: str, database_id: Optional[str]):
                self.project_id = project_id
                self.database_id = database_id if database_id is not None else "(default)"

            async def _get_access_token(self) -> str:
                try:
                    credentials, _ = await asyncio.to_thread(google.auth.default)
                    if not credentials.valid:
                        request = google.auth.transport.requests.Request()
                        await asyncio.to_thread(credentials.refresh, request)
                    return credentials.token
                except Exception as e:
                    raise Exception(f"Failed to get access token: {e}")

            async def create_and_wait_for_index(
                self, kind: str, timeout_seconds: int = 600
            ):
                print(f"Attempting to create index for kind: {kind}...") 
                try:
                    token = await self._get_access_token()
                    uri_create = f"https://firestore.googleapis.com/v1/projects/{self.project_id}/databases/{self.database_id}/collectionGroups/{kind}/indexes?alt=json"
                    headers = {
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    }
                    index_body = {
                        "apiScope": "DATASTORE_MODE_API",
                        "fields": [{"fieldPath": "timestamp", "order": "DESCENDING"}],
                        "queryScope": "COLLECTION_RECURSIVE",
                    }
                    response = await asyncio.to_thread(
                        requests.post, uri_create, headers=headers, data=json.dumps(index_body)
                    )

                    if response.status_code == 409:  # Index already exists
                        print(f"Index for kind '{kind}' already exists. Assuming it is serving.")
                        return
                    elif response.status_code != 200:
                        raise Exception(f"Failed to create index. Status: {response.status_code}, Response: {response.text}")

                    response_json = response.json()
                    operation_name = response_json.get("name")
                    if not operation_name:
                        raise Exception(f"No operation name in response: {response.text}")

                    print(f"Index creation operation started: {operation_name}")
                    print(f"This usually takes a couple of minutes")
                    await self._poll_operation_status(token, operation_name, timeout_seconds)

                except Exception as e:
                    print(f"An unexpected error during index creation: {e}")
                    raise

            async def _poll_operation_status(self, token: str, operation_name: str, timeout_seconds: int):
                start_time = time.time()
                while time.time() - start_time < timeout_seconds:
                    uri_poll = f"https://firestore.googleapis.com/v1/{operation_name}?alt=json"
                    headers = {
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json"
                    }
                    response = await asyncio.to_thread(requests.get, uri_poll, headers=headers)

                    if response.status_code != 200:
                        raise Exception(f"Failed to get operation status. Status: {response.status_code}, Response: {response.text}")

                    response_json = response.json()
                    if response_json.get("done"):
                        if "error" in response_json:
                            raise Exception(f"Index creation failed: {response_json['error']}")
                        print("Index creation operation completed. Index is now SERVING.")
                        return
                    else:
                        elapsed = int(time.time() - start_time)
                        print(f"Index creation in progress... Time elapsed: {elapsed} seconds.")

                    await asyncio.sleep(10) # Wait before polling again
                raise TimeoutError(f"Index did not become SERVING within {timeout_seconds} seconds.")

        index_manager = DatastoreIndexManager(project_id=self.client.project, database_id=self.client.database)
        await index_manager.create_and_wait_for_index(kind_name)

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        if state is None:
            state = {}

        def _transactional_create():
            with self.client.transaction() as transaction:
                app_key = self.key_factory.create_app_key(app_name)
                user_key = self.key_factory.create_user_key(app_name, user_id)

                entities_to_get = [app_key, user_key]
                retrieved_entities = self.client.get_multi(
                    entities_to_get, transaction=transaction
                )
                entity_map = {e.key: e for e in retrieved_entities}
                app_entity = entity_map.get(app_key)
                user_entity = entity_map.get(user_key)

                if not app_entity:
                    app_entity = datastore.Entity(key=app_key, exclude_from_indexes=("state",))
                    app_entity["state"] = {}
                if not user_entity:
                    user_entity = datastore.Entity(key=user_key, exclude_from_indexes=("state",))
                    user_entity["state"] = {}

                app_state_delta, user_state_delta, session_state = _extract_state_delta(
                    state
                )

                if app_state_delta:
                    app_entity["state"].update(app_state_delta)
                if user_state_delta:
                    user_entity["state"].update(user_state_delta)

                session_key = self.key_factory.create_session_key(
                    app_name, user_id, session_id
                )
                session_entity = datastore.Entity(key=session_key, exclude_from_indexes=("state",))
                session_entity.update(
                    {
                        "app_name": app_name,
                        "user_id": user_id,
                        "id": session_key.name,
                        "state": session_state,
                        "create_time": datetime.utcnow(),
                        "update_time": datetime.utcnow(),
                    }
                )

                entities_to_put = [app_entity, user_entity, session_entity]
                for entity in entities_to_put:
                    transaction.put(entity)

                merged_state = _merge_state(
                    app_entity["state"], user_entity["state"], session_entity["state"]
                )

                return Session(
                    app_name=app_name,
                    user_id=user_id,
                    id=session_key.name,
                    state=merged_state,
                    events=[],
                    last_update_time=session_entity["update_time"].timestamp(),
                )
        return await asyncio.to_thread(_transactional_create)

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        app_key = self.key_factory.create_app_key(app_name)
        user_key = self.key_factory.create_user_key(app_name, user_id)
        session_key = self.key_factory.create_session_key(
            app_name, user_id, session_id
        )

        entities_to_get = [app_key, user_key, session_key]
        retrieved_entities = await asyncio.to_thread(self.client.get_multi, entities_to_get)

        entity_map = {e.key: e for e in retrieved_entities}
        app_entity = entity_map.get(app_key)
        user_entity = entity_map.get(user_key)
        session_entity = entity_map.get(session_key)

        if not session_entity:
            return None

        storage_events = await self._get_session_events_with_index_handling(
            session_key, config
        )
        events = [self._entity_to_event(e) for e in reversed(storage_events)]

        app_state = app_entity["state"] if app_entity else {}
        user_state = user_entity["state"] if user_entity else {}
        session_state = session_entity["state"]

        merged_state = _merge_state(app_state, user_state, session_state)

        return Session(
            app_name=session_entity["app_name"],
            user_id=session_entity["user_id"],
            id=session_entity["id"],
            state=merged_state,
            events=events,
            last_update_time=session_entity["update_time"].timestamp(),
        )

    async def _get_session_events_with_index_handling(
        self,
        session_key,
        config: Optional[GetSessionConfig],
    ):
        """Fetches events and handles missing index errors by creating the index and retrying."""
        query = self.client.query(
            kind=self.key_factory.event_prefix, ancestor=session_key
        )
        query.order = ["-timestamp"]
        limit = config.num_recent_events if config and config.num_recent_events else None

        try:
            storage_events = await asyncio.to_thread(list, query.fetch(limit=limit))
            return storage_events
        except FailedPrecondition as e:
            if "400 no matching index found" in str(e).lower(): # this must be 400 no matching index found
                print(
                    "Query failed due to a missing index. Attempting to create it and wait..."
                )
                await self._handle_missing_index(self.key_factory.event_prefix)
                
                # After waiting, retry the query.
                print("Index is ready. Retrying the query...")
                storage_events = await asyncio.to_thread(list, query.fetch(limit=limit))
                return storage_events
            else:
                raise

    async def list_sessions(
        self, *, app_name: str, user_id: str
    ) -> ListSessionsResponse:
        user_key = self.key_factory.create_user_key(app_name, user_id)
        app_key = self.key_factory.create_app_key(app_name)

        query = self.client.query(
            kind=self.key_factory.session_prefix, ancestor=user_key
        )
        results = await asyncio.to_thread(list, query.fetch())

        retrieved_entities = await asyncio.to_thread(self.client.get_multi, [app_key, user_key])
        entity_map = {e.key: e for e in retrieved_entities}
        app_entity = entity_map.get(app_key)
        user_entity = entity_map.get(user_key)

        app_state = app_entity["state"] if app_entity else {}
        user_state = user_entity["state"] if user_entity else {}

        sessions = []
        for storage_session in results:
            session_state = storage_session["state"]
            merged_state = _merge_state(app_state, user_state, session_state)
            sessions.append(
                Session(
                    app_name=storage_session["app_name"],
                    user_id=storage_session["user_id"],
                    id=storage_session["id"],
                    state=merged_state,
                    events=[],
                    last_update_time=storage_session["update_time"].timestamp(),
                )
            )
        return ListSessionsResponse(sessions=sessions)

    async def delete_session(
        self, app_name: str, user_id: str, session_id: str
    ) -> None:
        session_key = self.key_factory.create_session_key(
            app_name, user_id, session_id
        )

        event_query = self.client.query(
            kind=self.key_factory.event_prefix, ancestor=session_key
        )
        event_query.keys_only()
        event_keys = await asyncio.to_thread(list, event_query.fetch())
        keys_to_delete = [entity.key for entity in event_keys] + [session_key]

        await asyncio.to_thread(self.client.delete_multi, keys_to_delete)

    async def append_event(self, session: Session, event: Event) -> Event:
        if event.partial:
            return event

        def _transactional_append():
            with self.client.transaction() as transaction:
                app_key = self.key_factory.create_app_key(session.app_name)
                user_key = self.key_factory.create_user_key(
                    session.app_name, session.user_id
                )
                session_key = self.key_factory.create_session_key(
                    session.app_name, session.user_id, session.id
                )

                entities_to_get = [app_key, user_key, session_key]
                retrieved_entities = self.client.get_multi(
                    entities_to_get, transaction=transaction
                )
                entity_map = {e.key: e for e in retrieved_entities}
                app_entity = entity_map.get(app_key)
                user_entity = entity_map.get(user_key)
                session_entity = entity_map.get(session_key)

                if not session_entity:
                    raise ValueError(f"Session with id {session.id} not found.")

                app_state_delta, user_state_delta, session_state_delta = {}, {}, {}
                if event.actions and event.actions.state_delta:
                    (
                        app_state_delta,
                        user_state_delta,
                        session_state_delta,
                    ) = _extract_state_delta(event.actions.state_delta)

                if app_state_delta and app_entity:
                    app_entity["state"].update(app_state_delta)
                if user_state_delta and user_entity:
                    user_entity["state"].update(user_state_delta)
                if session_state_delta:
                    session_entity["state"].update(session_state_delta)

                session_entity["update_time"] = datetime.utcnow()

                event_entity = self._event_to_entity(session, event)

                entities_to_put = [app_entity, user_entity, session_entity, event_entity]
                for entity in [e for e in entities_to_put if e is not None]:
                    transaction.put(entity)

                session.last_update_time = session_entity["update_time"].timestamp()

        await asyncio.to_thread(_transactional_append)
        await super().append_event(session=session, event=event)
        return event

    def _event_to_entity(self, session: Session, event: Event) -> datastore.Entity:
        """Converts an Event object to a Datastore entity."""
        event_key = self.key_factory.create_event_key(
            session.app_name, session.user_id, session.id, event.id
        )
        entity = datastore.Entity(
            key=event_key,
            exclude_from_indexes=(
                "actions",
                "content",
                "grounding_metadata",
                "custom_metadata",
            ),
        )
        # Serialize the content object to a JSON string if it exists
        content_for_datastore = None
        if event.content:
            content_dict = event.content.model_dump(exclude_none=True, mode="json")
            if self.strip_large_content and "parts" in content_dict:
                for part in content_dict["parts"]:
                    if "inline_data" in part and "data" in part["inline_data"]:
                        part["inline_data"]["data"] = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
            json_bytes = json.dumps(content_dict).encode("utf-8")
            content_for_datastore = gzip.compress(json_bytes)
        entity.update(
            {
                "id": event.id,
                "invocation_id": event.invocation_id,
                "author": event.author,
                "branch": event.branch,
                "actions": event.actions.model_dump(exclude_none=True, mode="json")
                if event.actions
                else None,
                "timestamp": datetime.fromtimestamp(event.timestamp),
                "long_running_tool_ids_json": list(event.long_running_tool_ids or []),
                "partial": event.partial,
                "turn_complete": event.turn_complete,
                "error_code": event.error_code,
                "error_message": event.error_message,
                "interrupted": event.interrupted,
                "content": content_for_datastore,
                "grounding_metadata": event.grounding_metadata.model_dump(
                    exclude_none=True, mode="json"
                )
                if event.grounding_metadata
                else None,
                "custom_metadata": event.custom_metadata,
            }
        )
        return entity

    def _entity_to_event(self, entity: datastore.Entity) -> Event:
        """Converts a Datastore entity to an Event object."""
        from google.cloud.datastore.entity import Entity
        actions_dict = entity.get("actions")
        actions = EventActions(**actions_dict) if actions_dict else None
        
        # --- MODIFICATION: Handles bytes (new), Entity (old), and string (transitional) ---
        content_val = entity.get("content")
        content_dict = None

        if isinstance(content_val, bytes):
            # It's a byte blob. Decode and parse from JSON.
            try:
                # Assume it's gzipped. Decompress, then decode.
                decompressed_bytes = gzip.decompress(content_val)
                content_dict = json.loads(decompressed_bytes.decode("utf-8"))
            except (gzip.BadGzipFile, OSError):
                # It's not gzipped, so it must be our old plain bytes.
                content_dict = json.loads(content_val.decode("utf-8"))
        elif isinstance(content_val, (Entity, dict)):
            # It's a nested Entity. Convert to a plain dict.
            content_dict = dict(content_val)
        elif isinstance(content_val, str):
            # It's a string. Parse from JSON.
            content_dict = json.loads(content_val)

        return Event(
            id=entity["id"],
            invocation_id=entity["invocation_id"],
            author=entity["author"],
            branch=entity["branch"],
            actions=actions,
            timestamp=entity["timestamp"].timestamp(),
            content=_session_util.decode_content(content_dict),
            long_running_tool_ids=set(entity.get("long_running_tool_ids_json", [])),
            partial=entity.get("partial"),
            turn_complete=entity.get("turn_complete"),
            error_code=entity.get("error_code"),
            error_message=entity.get("error_message"),
            interrupted=entity.get("interrupted"),
            grounding_metadata=_session_util.decode_grounding_metadata(
                entity.get("grounding_metadata")
            ),
            custom_metadata=entity.get("custom_metadata"),
        )
    
    def _debug_print_entity_details(self, entity: datastore.Entity):
        """Prints the detailed properties and sizes for a Datastore entity."""
        if not entity:
            print("--- Skipping None entity ---")
            return
        from sys import getsizeof
        print("\n--- Detailed Entity Debug ---")
        print(f"Entity Key: {entity.key}")
        print("Properties:")
        total_size = 0
        try:
            for key, value in entity.items():
                size = 0
                # Calculate size based on type
                if isinstance(value, str):
                    size = len(value.encode("utf-8"))
                elif isinstance(value, bytes):
                    size = len(value)
                elif isinstance(value, (dict, list)):
                    # Simulating JSON size, as Datastore stores it this way
                    size = len(json.dumps(value).encode("utf-8"))
                else:
                    # A rough estimate for other types like numbers, datetime
                    size = getsizeof(value)

                total_size += size
                print(f"  - Property: '{key}', Type: {type(value).__name__}, Size: {size} bytes")

                if size > 1500:
                    print(f"  !!!! DANGER: Property '{key}' is > 1500 bytes and will cause an error if indexed.")

            print(f"Total Estimated Entity Size: {total_size} bytes")
            if total_size > 1048576: # 1 MiB limit
                 print("  !!!! DANGER: Total entity size exceeds the 1 MiB limit.")
            print("--- End Detailed Entity Debug ---")

            # Also print the raw dictionary to see the structure
            print("--- Raw Entity Dictionary ---")
            # Create a serializable dictionary representation
            raw_dict = {}
            for k, v in entity.items():
                if isinstance(v, datetime):
                    raw_dict[k] = v.isoformat()
                else:
                    raw_dict[k] = v
            print(json.dumps(raw_dict, indent=2))
            print("--- End Raw Entity Dictionary ---")

        except Exception as e:
            print(f"An error occurred during debugging print: {e}")

def _extract_state_delta(state: dict[str, Any]):
    app_state_delta = {}
    user_state_delta = {}
    session_state_delta = {}
    if state:
        for key, value in state.items():
            if key.startswith(State.APP_PREFIX):
                app_state_delta[key.removeprefix(State.APP_PREFIX)] = value
            elif key.startswith(State.USER_PREFIX):
                user_state_delta[key.removeprefix(State.USER_PREFIX)] = value
            elif not key.startswith(State.TEMP_PREFIX):
                session_state_delta[key] = value
    return app_state_delta, user_state_delta, session_state_delta


def _merge_state(app_state, user_state, session_state):
    merged_state = copy.deepcopy(session_state)
    if app_state:
        for key, value in app_state.items():
            merged_state[State.APP_PREFIX + key] = value
    if user_state:
        for key, value in user_state.items():
            merged_state[State.USER_PREFIX + key] = value
    return merged_state