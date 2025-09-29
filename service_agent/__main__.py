from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError
from common.utils.push_notification_auth import PushNotificationSenderAuth
from task_manager import AgentTaskManager
from .agent import ServiceAgent
import click
import os
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10000)

def main(host, port):
    """Starts the Services Agent server."""
    try:
        # if not os.getenv("GOOGLE_API_KEY"):
        #     raise MissingAPIKeyError("GOOGLE_API_KEY environment variable not set.")
        
        # Mô tả agent hỗ trợ gì
        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)

        skill = AgentSkill(
            id="health_service_selection",
            name="Health Service Recommendation",
            description="Recommend health services to users base on conversation",
            tags=["health", "service", "recommendation"],
            examples=[
                "Tôi muốn kiểm tra tổng quát sức khỏe.",
                "Có gói khám tim mạch nào không?",
                "Tư vấn giúp tôi chọn gói khám phù hợp với người lớn tuổi."
            ],
        )
        agent_card = AgentCard(
            name="Service Agent",
            description="Help users choose suitable medical services",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=ServiceAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=ServiceAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # đối tượng quản lý bảo mật cho việc gửi notification từ agent → client.
        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=ServiceAgent(), notification_sender_auth=notification_sender_auth),
            host=host,
            port=port,
        )

        server.app.add_route(
            "/.well-known/jwks.json", notification_sender_auth.handle_jwks_endpoint, methods=["GET"]
        )

        logger.info(f"Starting server on {host}:{port}")
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)


if __name__ == "__main__":
    main()