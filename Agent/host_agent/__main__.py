import logging
import os

import click
import uvicorn

from dotenv import load_dotenv
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from common.server import A2AServer
from common.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from host_agent_executor import HostAgentExecutor
from routing_agent import root_agent
from task_manager import ADKTaskManager  # Import ADKTaskManager instead

load_dotenv()
logging.basicConfig()

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8083


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    # Verify an API key is set
    if os.getenv('GOOGLE_GENAI_USE_VERTEXAI') != 'TRUE' and not os.getenv('GOOGLE_API_KEY'):
        raise ValueError('GOOGLE_API_KEY not found')

    app_url = os.environ.get('APP_URL', f'http://{host}:{port}')

    capabilities = AgentCapabilities(
        streaming=True, 
        pushNotifications=True,
        stateTransitionHistory=True
    )

    agent_card = AgentCard(
        name='Host A2A Agent',
        description='Multi-agent orchestrator for medical task delegation',
        url=app_url,
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=capabilities,
        skills=[
            AgentSkill(
                id='host_agent',
                name='Medical Orchestrator',
                description='Routes medical queries to specialized agents',
                tags=['orchestration', 'medical', 'routing'],
                examples=['I have a headache', 'Book appointment', 'Cost inquiry'],
            )
        ],
    )

    adk_agent = root_agent
    runner = Runner(
        app_name=agent_card.name,
        agent=adk_agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )
    
    agent_executor = HostAgentExecutor(runner, agent_card)
    
    # Use ADKTaskManager instead of InMemoryTaskManager
    task_manager = ADKTaskManager(agent_executor)

    a2a_server = A2AServer(
        host=host,
        port=port,
        agent_card=agent_card,
        task_manager=task_manager,
    )
    
    a2a_server.start()


@click.command()
@click.option('--host', 'host', default=DEFAULT_HOST)
@click.option('--port', 'port', default=DEFAULT_PORT)
def cli(host: str, port: int):
    main(host, port)


if __name__ == '__main__':
    cli()