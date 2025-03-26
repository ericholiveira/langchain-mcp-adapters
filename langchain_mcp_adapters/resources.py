from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from mcp import ClientSession
from pydantic import BaseModel
from mcp.types import Resource,ResourceTemplate

from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextResourceContents,
)

NonTextContent = ImageContent | EmbeddedResource


def _convert_call_tool_result(
    call_tool_result: CallToolResult,
) -> tuple[str | list[str], list[NonTextContent] | None]:
    text_contents: list[TextResourceContents] = []
    non_text_contents = []
    for content in call_tool_result.contents:
        if isinstance(content, TextResourceContents):
            text_contents.append(content)
        else:
            non_text_contents.append(content)
    tool_content: str | list[str] = [content.text for content in text_contents]
    if len(text_contents) == 1:
        tool_content = tool_content[0]
    return tool_content, non_text_contents or None

class ResourceSchema(BaseModel):
    resource_uri:str


async def convert_resources_to_langchain_tool(session: ClientSession) -> BaseTool:
    """Creates a tool from the MCP resources to make it controlled by the LLM.

    NOTE: this tool can be executed only in a context of an active MCP client session.

    Args:
        session: MCP client session

    Returns:
        a LangChain tool
    """
    resources = await list_mcp_resources(session)
    resources = [{"uriTemplate": r.uri, "name":r.name,"description":r.description,"mimeType":r.mimeType,"size":r.size,"annotations":r.annotations} for r in resources]
    resource_templates = await list_mcp_resource_templates(session)
    resource_templates = [{"uriTemplate": r.uriTemplate, "name":r.name,"description":r.description,"mimeType":r.mimeType,"annotations":r.annotations} for r in resource_templates]
    resources = resources + resource_templates
    resource_tool_description = f"""
    This tool returns the contents of a MCP resource, from a list of available resource.
    The list of available resources are:
    {'\n'.join([str(r) for r in resources])}
    """
    async def call_tool(
        **arguments: dict[str, Any],
    )-> str:
        call_tool_result = await session.read_resource(uri=arguments["resource_uri"])
        return _convert_call_tool_result(call_tool_result)
    
    return StructuredTool(
        name="get_mcp_resource",
        description=resource_tool_description,
        args_schema=ResourceSchema.schema(),
        coroutine=call_tool,
        response_format="content_and_artifact",
    )

async def list_mcp_resource_templates(session: ClientSession) -> list[ResourceTemplate]:
    """List all MCP resources templates available in the server."""
    resource_templates = await session.list_resource_templates()
    return resource_templates.resourceTemplates

async def list_mcp_resources(session: ClientSession) -> list[Resource]:
    """List all MCP resources available in the server."""
    resources = await session.list_resources()
    return resources.resources
