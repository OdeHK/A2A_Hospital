"""
Adapter để kết nối HostAgentExecutor với InMemoryTaskManager.
Adapter tích hợp ADK Agent Executor với Common types system.
Refactored version với type hints, better error handling, extensibility.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from typing import AsyncIterable, Union, Optional, List

from common.server import InMemoryTaskManager
from common.types import (
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    TaskStatus,
    TaskState,
    Message,
    JSONRPCResponse,
    Part,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class RequestContextAdapter:
    """Wrapper để chuyển đổi Common types request thành ADK RequestContext."""

    def __init__(self, request):
        self.task_id = request.params.id
        self.context_id = request.params.sessionId
        self.message = request.params.message
        self.current_task = None
        self.requested_extensions = []
        self._activated_extensions = []

    def add_activated_extension(self, extension_uri: str):
        self._activated_extensions.append(extension_uri)

    @property
    def activated_extensions(self):
        return self._activated_extensions


class EventQueueAdapter:
    """
    Adapter để chuyển đổi ADK events sang Common types events.
    """

    def __init__(self, task_manager: 'ADKTaskManager', task_id: str):
        self.task_manager = task_manager
        self.task_id = task_id

    async def push(self, event):
        """
        Nhận event từ ADK và chuyển sang Common types format.
        """
        # Chuyển đổi event
        if hasattr(event, 'status'):
            common_event = TaskStatusUpdateEvent(
                id=self.task_id,
                status=event.status,
                final=getattr(event, 'final', False),
            )
        elif hasattr(event, 'artifact'):
            common_event = TaskArtifactUpdateEvent(
                id=self.task_id,
                artifact=event.artifact,
            )
        else:
            logger.warning(f"Unknown event type: {type(event)}")
            return

        # Push vào SSE queue
        await self.task_manager.enqueue_events_for_sse(self.task_id, common_event)

        # Update store - FIX: Không gọi update_store với status=None
        if isinstance(common_event, TaskStatusUpdateEvent):
            await self.task_manager.update_store(
                self.task_id,
                common_event.status,
                []
            )
        elif isinstance(common_event, TaskArtifactUpdateEvent):
            # Chỉ update artifacts, không touch status
            async with self.task_manager.lock:
                task = self.task_manager.tasks.get(self.task_id)
                if task:
                    if task.artifacts is None:
                        task.artifacts = []
                    task.artifacts.append(common_event.artifact)


class ADKTaskManager(InMemoryTaskManager):
    """
    TaskManager adapter kết nối ADK HostAgentExecutor với Common types.
    """

    def __init__(self, agent_executor):
        super().__init__()
        self.agent_executor = agent_executor

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Xử lý send task request từ client.
        """
        logger.info(f"Handling send_task for {request.params.id}")
        
        # Upsert task vào store
        task = await self.upsert_task(request.params)
        
        # Tạo context cho agent executor
        context = RequestContextAdapter(request)
        
        # Tạo event queue để nhận updates
        event_queue = EventQueueAdapter(self, request.params.id)
        
        try:
            # Execute agent
            await self.agent_executor.execute(context, event_queue)
            
            # Lấy task đã hoàn thành
            final_task = await self.tasks.get(request.params.id)
            
            return SendTaskResponse(id=request.id, result=final_task)
        
        except Exception as e:
            logger.error(f"Error executing task: {e}", exc_info=True)
            await self.update_store(
                request.params.id,
                TaskStatus(
                    state=TaskState.FAILED, 
                    message=Message(
                        role="agent",
                        parts=[{"type": "text", "text": f"Error: {str(e)}"}]
                    )
                ),
                []
            )
            raise

    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        """
        Xử lý streaming send task request.
        
        IMPORTANT: Phương thức này phải return một async generator,
        KHÔNG phải await nó. Server sẽ check isinstance(result, AsyncIterable)
        và xử lý streaming response.
        """
        logger.info(f"Handling send_task_subscribe for {request.params.id}")
        
        # Return async generator directly (không await)
        return self._streaming_generator(request)
    
    async def _streaming_generator(
        self, request: SendTaskStreamingRequest
    ):
        """
        Async generator thực tế cho streaming response.
        """
        try:
            # Setup SSE consumer
            sse_queue = await self.setup_sse_consumer(request.params.id)
            
            # Upsert task
            await self.upsert_task(request.params)
            
            # Tạo context và event queue
            context = RequestContextAdapter(request)
            event_queue = EventQueueAdapter(self, request.params.id)
            
            # Start agent execution in background
            async def execute_agent():
                try:
                    await self.agent_executor.execute(context, event_queue)
                except Exception as e:
                    logger.error(f"Agent execution error: {e}", exc_info=True)
            
            execution_task = asyncio.create_task(execute_agent())
            
            # Stream events as they arrive
            try:
                async for event in self.dequeue_events_for_sse(
                    request.id, 
                    request.params.id, 
                    sse_queue
                ):
                    yield event
            finally:
                # Cleanup
                if not execution_task.done():
                    execution_task.cancel()
                    try:
                        await execution_task
                    except asyncio.CancelledError:
                        logger.debug("Agent execution cancelled")
                        
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            raise