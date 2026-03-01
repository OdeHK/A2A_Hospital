import json
import logging

from typing import TYPE_CHECKING

from common.types import (
    AgentCard,
    FilePart,
    FileContent,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
    TaskStatus,
    Message,
    DataPart,
)

from google.adk import Runner
from google.genai import types


if TYPE_CHECKING:
    from google.adk.sessions.session import Session


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_USER_ID = 'self'


class HostAgentExecutor:
    """An AgentExecutor that runs an ADK-based Agent for host_agent."""

    def __init__(self, runner: Runner, card: AgentCard):
        self.runner = runner
        self._card = card
        self._active_sessions: set[str] = set()

    async def _process_request(self, new_message: types.Content, session_id: str, task_updater,) -> None:
        session_obj = await self._upsert_session(session_id)
        session_id = session_obj.id

        self._active_sessions.add(session_id)

        try:
            async for event in self.runner.run_async(
                user_id=DEFAULT_USER_ID,
                session_id=session_id,
                new_message=new_message,
            ):
                logger.debug(
                    '### Event received: %s',
                    event.model_dump_json(exclude_none=True, indent=2),
                )

                if event.is_final_response():
                    parts = [
                        convert_genai_part_to_common(part)
                        for part in event.content.parts
                        if (part.text or part.file_data or part.inline_data)
                    ]

                    # If no valid parts found, try to extract from thought_signature or other metadata
                    if not parts and event.content.parts:
                        logger.debug('No standard parts found, checking for alternative content')
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                parts.append(TextPart(text=part.text))

                    logger.debug('#### Yielding final response: %s', parts)
                    await task_updater.add_artifact(parts)

                    await task_updater.update_status(
                        TaskState.COMPLETED, final=True
                    )

                    break
                    
                if not event.get_function_calls():
                    parts = [
                        convert_genai_part_to_common(part)
                        for part in event.content.parts
                        if (part.text or part.file_data or part.inline_data)
                    ]

                    # Filter out empty parts
                    parts = [p for p in parts if p is not None]
                    
                    if not parts:
                        logger.debug('No parts found in update event, skipping')
                        continue

                    logger.debug('#### Yielding update response: %s', parts)
                    await task_updater.update_status(
                        TaskState.WORKING,
                        message=task_updater.new_agent_message(parts),
                    )
                else:
                    logger.debug('#### Event - Function Calls')

        finally:
            self._active_sessions.discard(session_id)

    async def execute(self, context, event_queue,):
        logger.debug('[host_agent] execute called with context: %s', context)

        updater = TaskUpdaterAdapter(event_queue, context.task_id, context.context_id)
        
        if not context.current_task:
            await updater.update_status(TaskState.SUBMITTED)
        await updater.update_status(TaskState.WORKING)
        
        await self._process_request(
            types.UserContent(
                parts=[
                    convert_common_part_to_genai(part)
                    for part in context.message.parts
                ],
            ),
            context.context_id,
            updater,
        )
        logger.debug('[host_agent] execute exiting')

    async def cancel(self, context, event_queue):
        """Cancel the execution for the given context."""
        session_id = context.context_id
        if session_id in self._active_sessions:
            logger.info(
                f'Cancellation requested for active host_agent session: {session_id}'
            )
            self._active_sessions.discard(session_id)
        else:
            logger.debug(
                f'Cancellation requested for inactive host_agent session: {session_id}'
            )

        raise ValueError(UnsupportedOperationError())

    async def _upsert_session(self, session_id: str) -> 'Session':
        """Retrieves a session if it exists, otherwise creates a new one."""
        session = await self.runner.session_service.get_session(
            app_name=self.runner.app_name,
            user_id=DEFAULT_USER_ID,
            session_id=session_id,
        )
        if session is None:
            session = await self.runner.session_service.create_session(
                app_name=self.runner.app_name,
                user_id=DEFAULT_USER_ID,
                session_id=session_id,
            )
        return session


class TaskUpdaterAdapter:
    """Adapter để chuyển đổi calls sang EventQueueAdapter."""

    def __init__(self, event_queue, task_id: str, context_id: str):
        self.event_queue = event_queue
        self.task_id = task_id
        self.context_id = context_id

    async def update_status(self, state: TaskState, message: Message = None, final: bool = False):
        """Update task status."""
        from common.types import TaskStatusUpdateEvent, TaskStatus
        
        event = TaskStatusUpdateEvent(
            id=self.task_id,
            status=TaskStatus(state=state, message=message),
            final=final,
        )
        await self.event_queue.push(event)

    async def add_artifact(self, parts: list[Part]):
        """Add artifact with parts."""
        from common.types import TaskArtifactUpdateEvent, Artifact
        
        event = TaskArtifactUpdateEvent(
            id=self.task_id,
            artifact=Artifact(parts=parts),
        )
        await self.event_queue.push(event)

    def new_agent_message(self, parts: list[Part]) -> Message:
        """Create new agent message."""
        return Message(role="agent", parts=parts)


def convert_common_part_to_genai(part: Part) -> types.Part:
    """Convert Common Part to Google Gen AI Part."""
    if isinstance(part, TextPart):
        return types.Part(text=part.text)
    if isinstance(part, FilePart):
        file = part.file
        if file.uri:
            return types.Part(
                file_data=types.FileData(
                    file_uri=file.uri, 
                    mime_type=file.mimeType
                )
            )
        if file.bytes:
            return types.Part(
                inline_data=types.Blob(
                    data=file.bytes, 
                    mime_type=file.mimeType
                )
            )
        raise ValueError(f'File must have uri or bytes')
    if isinstance(part, DataPart):
        return types.Part(text=json.dumps(part.data))
    raise ValueError(f'Unsupported part type: {type(part)}')


def convert_genai_part_to_common(part: types.Part) -> Part:
    """Convert Google Gen AI Part to Common Part."""
    if part.text:
        return TextPart(text=part.text)
    if part.file_data:
        return FilePart(
            file=FileContent(
                uri=part.file_data.file_uri,
                mimeType=part.file_data.mime_type,
            )
        )
    if part.inline_data:
        return FilePart(
            file=FileContent(
                bytes=part.inline_data.data,
                mimeType=part.inline_data.mime_type,
            )
        )
    return None