import logging
from collections.abc import Callable

import httpx

from common.client import A2AClient
from common.types import (
    AgentCard,
    TaskSendParams,
    SendTaskResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)

from dotenv import load_dotenv


load_dotenv()

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, agent_card: AgentCard, agent_url: str = None):
        logger.debug(f'agent_card: {agent_card}')
        
        if agent_url is None:
            agent_url = agent_card.url
            
        logger.debug(f'agent_url: {agent_url}')
        
        # Tạo httpx client để quản lý connections
        self._httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0)
        )
        
        # Truyền httpx_client vào A2AClient nếu nó support
        # Nếu không, A2AClient sẽ tự tạo client riêng
        self.agent_client = A2AClient(agent_card=agent_card, url=agent_url)
        
        # Inject httpx client vào A2AClient nếu có attribute
        if hasattr(self.agent_client, '_client'):
            self.agent_client._client = self._httpx_client
        
        self.card = agent_card

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_task(
        self, 
        task_params: TaskSendParams
    ) -> SendTaskResponse:
        """
        Gửi task tới remote agent.
        
        Args:
            task_params: TaskSendParams object chứa thông tin task
        
        Returns:
            SendTaskResponse object
        """
        payload = task_params.model_dump(exclude_none=True)
        return await self.agent_client.send_task(payload)
    
    async def send_task_streaming(
        self,
        task_params: TaskSendParams
    ):
        """
        Gửi task với streaming support.
        
        Args:
            task_params: TaskSendParams object
        
        Yields:
            SendTaskStreamingResponse objects
        """
        payload = task_params.model_dump(exclude_none=True)
        async for response in self.agent_client.send_task_streaming(payload):
            yield response
    
    async def close(self):
        """Đóng tất cả connections"""
        try:
            await self._httpx_client.aclose()
            logger.debug(f"Closed HTTP client for {self.card.name}")
        except Exception as e:
            logger.error(f"Error closing HTTP client: {e}")
    
    async def __aenter__(self):
        """Support async context manager"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context"""
        await self.close()
        return False