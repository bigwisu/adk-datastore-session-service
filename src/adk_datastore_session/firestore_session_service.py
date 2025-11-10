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

import firebase_admin
from firebase_admin import credentials, firestore
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import _session_util
from google.adk.sessions.base_session_service import (BaseSessionService,
                                                      GetSessionConfig,
                                                      ListSessionsResponse)
from google.adk.sessions.session import Session
from google.adk.sessions.state import State


class FirestoreCollectionNames:
    """Defines the collection names used in Firestore."""

    def __init__(
        self,
        app_collection="ADKApps",
        user_collection="ADKUsers",
        session_collection="ADKSessions",
        event_collection="ADKEvents",
    ):
        self.app_collection = app_collection
        self.user_collection = user_collection
        self.session_collection = session_collection
        self.event_collection = event_collection


class FirestorePaths:
    """Functions for creating Firestore document paths."""

    def __init__(self, db, collection_names: Optional[FirestoreCollectionNames] = None):
        self.db = db
        if collection_names is None:
            collection_names = FirestoreCollectionNames()
        self.collections = collection_names

    def app_ref(self, app_name: str):
        """Gets a DocumentReference for an App."""
        return self.db.collection(self.collections.app_collection).document(app_name)

    def user_ref(self, app_name: str, user_id: str):
        """Gets a DocumentReference for a User."""
        return self.app_ref(app_name).collection(self.collections.user_collection).document(user_id)

    def session_ref(self, app_name: str, user_id: str, session_id: str):
        """Gets a DocumentReference for a Session."""
        return self.user_ref(app_name, user_id).collection(self.collections.session_collection).document(session_id)

    def event_ref(self, app_name: str, user_id: str, session_id: str, event_id: str):
        """Gets a DocumentReference for an Event."""
        return self.session_ref(app_name, user_id, session_id).collection(self.collections.event_collection).document(event_id)


class FirestoreSessionService(BaseSessionService):
    """A session service that uses Google Cloud Firestore for storage."""

    def __init__(
        self,
        project: Optional[str] = None,
        cred: Optional[credentials.Certificate] = None,
        database: str = "(default)",
        collection_names: Optional[FirestoreCollectionNames] = None,
        strip_large_content: bool = True,
    ):
        """Initializes the Firestore session service."""
        if not firebase_admin._apps:
            app = firebase_admin.initialize_app(cred, {"projectId": project})
        else:
            app = firebase_admin.get_app()

        self.db = firestore.client(app=app, database=database)
        self.paths = FirestorePaths(self.db, collection_names)
        self.strip_large_content = strip_large_content

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
        if session_id is None:
            session_id = str(uuid.uuid4())

        app_ref = self.paths.app_ref(app_name)
        user_ref = self.paths.user_ref(app_name, user_id)
        session_ref = self.paths.session_ref(app_name, user_id, session_id)

        app_state_delta, user_state_delta, session_state = _extract_state_delta(state)

        @firestore.async_transactional
        async def _transactional_create(transaction):
            app_doc = await app_ref.get(transaction=transaction)
            user_doc = await user_ref.get(transaction=transaction)

            if not app_doc.exists:
                transaction.set(app_ref, {"state": app_state_delta})
                app_current_state = app_state_delta
            else:
                app_current_state = app_doc.to_dict().get("state", {})
                if app_state_delta:
                    transaction.update(app_ref, {"state": firestore.firestore.SERVER_TIMESTAMP, **app_state_delta})
                    app_current_state.update(app_state_delta)

            if not user_doc.exists:
                transaction.set(user_ref, {"state": user_state_delta})
                user_current_state = user_state_delta
            else:
                user_current_state = user_doc.to_dict().get("state", {})
                if user_state_delta:
                    transaction.update(user_ref, {"state": firestore.firestore.SERVER_TIMESTAMP, **user_state_delta})
                    user_current_state.update(user_state_delta)

            session_data = {
                "app_name": app_name,
                "user_id": user_id,
                "id": session_id,
                "state": session_state,
                "create_time": firestore.firestore.SERVER_TIMESTAMP,
                "update_time": firestore.firestore.SERVER_TIMESTAMP,
            }
            transaction.set(session_ref, session_data)

            merged_state = _merge_state(app_current_state, user_current_state, session_state)

            return Session(
                app_name=app_name,
                user_id=user_id,
                id=session_id,
                state=merged_state,
                events=[],
                last_update_time=datetime.utcnow().timestamp(), # Approximate
            )

        return await _transactional_create(self.db.transaction())

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        app_ref = self.paths.app_ref(app_name)
        user_ref = self.paths.user_ref(app_name, user_id)
        session_ref = self.paths.session_ref(app_name, user_id, session_id)

        docs = await self.db.getAll([app_ref, user_ref, session_ref])
        doc_map = {d.reference.path: d for d in docs}

        app_doc = doc_map.get(app_ref.path)
        user_doc = doc_map.get(user_ref.path)
        session_doc = doc_map.get(session_ref.path)

        if not session_doc or not session_doc.exists:
            return None

        session_data = session_doc.to_dict()

        events_query = session_ref.collection(self.paths.collections.event_collection).order_by(
            "timestamp", direction=firestore.Query.DESCENDING
        )
        if config and config.num_recent_events:
            events_query = events_query.limit(config.num_recent_events)

        event_docs = [doc async for doc in events_query.stream()]
        events = [self._doc_to_event(e) for e in reversed(event_docs)]

        app_state = app_doc.to_dict().get("state", {}) if app_doc and app_doc.exists else {}
        user_state = user_doc.to_dict().get("state", {}) if user_doc and user_doc.exists else {}
        session_state = session_data.get("state", {})

        merged_state = _merge_state(app_state, user_state, session_state)

        return Session(
            app_name=session_data["app_name"],
            user_id=session_data["user_id"],
            id=session_data["id"],
            state=merged_state,
            events=events,
            last_update_time=session_data["update_time"].timestamp(),
        )

    async def list_sessions(
        self, *, app_name: str, user_id: str
    ) -> ListSessionsResponse:
        user_ref = self.paths.user_ref(app_name, user_id)
        app_ref = self.paths.app_ref(app_name)

        docs = await self.db.getAll([app_ref, user_ref])
        doc_map = {d.reference.path: d for d in docs}
        app_doc = doc_map.get(app_ref.path)
        user_doc = doc_map.get(user_ref.path)

        app_state = app_doc.to_dict().get("state", {}) if app_doc and app_doc.exists else {}
        user_state = user_doc.to_dict().get("state", {}) if user_doc and user_doc.exists else {}

        session_docs = [doc async for doc in user_ref.collection(self.paths.collections.session_collection).stream()]

        sessions = []
        for session_doc in session_docs:
            session_data = session_doc.to_dict()
            session_state = session_data.get("state", {})
            merged_state = _merge_state(app_state, user_state, session_state)
            sessions.append(
                Session(
                    app_name=session_data["app_name"],
                    user_id=session_data["user_id"],
                    id=session_data["id"],
                    state=merged_state,
                    events=[],
                    last_update_time=session_data["update_time"].timestamp(),
                )
            )
        return ListSessionsResponse(sessions=sessions)

    async def delete_session(
        self, app_name: str, user_id: str, session_id: str
    ) -> None:
        session_ref = self.paths.session_ref(app_name, user_id, session_id)
        event_docs = [doc async for doc in session_ref.collection(self.paths.collections.event_collection).stream()]

        batch = self.db.batch()
        for doc in event_docs:
            batch.delete(doc.reference)
        batch.delete(session_ref)
        await asyncio.to_thread(batch.commit)

    async def append_event(self, session: Session, event: Event) -> Event:
        if event.partial:
            return event

        app_ref = self.paths.app_ref(session.app_name)
        user_ref = self.paths.user_ref(session.app_name, session.user_id)
        session_ref = self.paths.session_ref(session.app_name, session.user_id, session.id)
        event_ref = self.paths.event_ref(session.app_name, session.user_id, session.id, event.id)

        @firestore.async_transactional
        async def _transactional_append(transaction):
            session_doc = await session_ref.get(transaction=transaction)
            if not session_doc.exists:
                raise ValueError(f"Session with id {session.id} not found.")

            app_state_delta, user_state_delta, session_state_delta = {}, {}, {}
            if event.actions and event.actions.state_delta:
                (
                    app_state_delta,
                    user_state_delta,
                    session_state_delta,
                ) = _extract_state_delta(event.actions.state_delta)

            if app_state_delta:
                transaction.update(app_ref, {f"state.{k}": v for k, v in app_state_delta.items()})
            if user_state_delta:
                transaction.update(user_ref, {f"state.{k}": v for k, v in user_state_delta.items()})
            if session_state_delta:
                transaction.update(session_ref, {f"state.{k}": v for k, v in session_state_delta.items()})

            transaction.update(session_ref, {"update_time": firestore.firestore.SERVER_TIMESTAMP})

            event_data = self._event_to_dict(event)
            transaction.set(event_ref, event_data)

        await _transactional_append(self.db.transaction())
        await super().append_event(session=session, event=event)
        return event

    def _event_to_dict(self, event: Event) -> dict:
        """Converts an Event object to a Firestore-compatible dictionary."""
        content_for_firestore = None
        if event.content:
            content_dict = event.content.model_dump(exclude_none=True, mode="json")
            if self.strip_large_content and "parts" in content_dict:
                for part in content_dict["parts"]:
                    if "inline_data" in part and "data" in part["inline_data"]:
                        part["inline_data"]["data"] = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
            json_bytes = json.dumps(content_dict).encode("utf-8")
            content_for_firestore = gzip.compress(json_bytes)

        return {
            "id": event.id,
            "invocation_id": event.invocation_id,
            "author": event.author,
            "branch": event.branch,
            "actions": event.actions.model_dump(exclude_none=True, mode="json") if event.actions else None,
            "timestamp": datetime.fromtimestamp(event.timestamp),
            "long_running_tool_ids_json": list(event.long_running_tool_ids or []),
            "partial": event.partial,
            "turn_complete": event.turn_complete,
            "error_code": event.error_code,
            "error_message": event.error_message,
            "interrupted": event.interrupted,
            "content": content_for_firestore,
            "grounding_metadata": event.grounding_metadata.model_dump(exclude_none=True, mode="json") if event.grounding_metadata else None,
            "custom_metadata": event.custom_metadata,
        }

    def _doc_to_event(self, doc) -> Event:
        """Converts a Firestore document to an Event object."""
        data = doc.to_dict()
        actions_dict = data.get("actions")
        actions = EventActions(**actions_dict) if actions_dict else None

        content_val = data.get("content")
        content_dict = None
        if isinstance(content_val, bytes):
            try:
                decompressed_bytes = gzip.decompress(content_val)
                content_dict = json.loads(decompressed_bytes.decode("utf-8"))
            except (gzip.BadGzipFile, OSError):
                content_dict = json.loads(content_val.decode("utf-8"))
        elif isinstance(content_val, str):
            content_dict = json.loads(content_val)

        return Event(
            id=data["id"],
            invocation_id=data["invocation_id"],
            author=data["author"],
            branch=data["branch"],
            actions=actions,
            timestamp=data["timestamp"].timestamp(),
            content=_session_util.decode_content(content_dict),
            long_running_tool_ids=set(data.get("long_running_tool_ids_json", [])),
            partial=data.get("partial"),
            turn_complete=data.get("turn_complete"),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            interrupted=data.get("interrupted"),
            grounding_metadata=_session_util.decode_grounding_metadata(
                data.get("grounding_metadata")
            ),
            custom_metadata=data.get("custom_metadata"),
        )

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