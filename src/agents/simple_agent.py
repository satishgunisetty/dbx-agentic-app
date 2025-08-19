from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from mlflow.models import ModelConfig

from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain.agents import create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
from unitycatalog.ai.core.databricks import DatabricksFunctionClient


class SimpleDatabricksAgent(ResponsesAgent):
    """
    A simple agent that uses the Databricks Agent framework to respond to queries.
    This agent is designed to be used with the Databricks platform and can handle
    various types of queries related to data processing and analysis.
    """

    def __init__(self):
        self.config = ModelConfig(
            development_config={
                "endpoint_name": "databricks-meta-llama-3-3-70b-instruct",
                "temperature": 0.01,
                "max_tokens": 1000,
                "system_prompt": "You are a helpful assistant that can execute Python code.",
                "tool_list": ["system.ai.python_exec"],
                "cluster_id": "0731-074422-htlsdnk3",
            }
        )
        self.agent = self._build_agent()

    def _build_agent(self):

        try:
            llm = ChatDatabricks(
                model=self.config.get("endpoint_name"),
                temperature=self.config.get("temperature"),
                max_tokens=self.config.get("max_tokens"),
            )

            # Initialize Unity Catalog client
            client = DatabricksFunctionClient(
                execution_mode="local", cluster_id=self.config.get("cluster_id")
            )
            set_uc_function_client(client)

            # Load tools with the client
            toolkit = UCFunctionToolkit(
                function_names=self.config.get("tool_list"), client=client
            )

            return create_tool_calling_agent(
                llm=llm,
                tools=toolkit.tools,
                prompt=self.config.get("system_prompt"),
            )

        except Exception as e:
            raise ValueError(f"Failed to build agent: {e}")

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        if not self.agent:
            raise ValueError("Agent is not initialized")

        try:
            messages = [
                HumanMessage(content=msg.content, role=msg.role)
                for msg in request.input
            ]

            response = self.agent.invoke({"messages": messages})

            output = []

            for msg in response["messages"]:
                if isinstance(msg, AIMessage):
                    output.append(
                        {
                            "type": "message",
                            "id": str(uuid.uuid4()),
                            "content": [{"type": "output_text", "text": msg.content}],
                            "role": "assistant",
                        }
                    )
                elif isinstance(msg, ToolMessage):
                    output.append(
                        {
                            "type": "message",
                            "id": str(uuid.uuid4()),
                            "content": [{"type": "output_text", "text": msg.tool_call}],
                            "role": "tool",
                            "tool_call_id": msg.tool_call_id,
                        }
                    )

            return ResponsesAgentResponse(output=output)

        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
