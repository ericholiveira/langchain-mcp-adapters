import pytest
from unittest.mock import AsyncMock, Mock
from mcp import ClientSession
from mcp.types import (
    ListResourcesResult,
    ListResourceTemplatesResult,
    Resource,
    ResourceTemplate
)
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.resources import (
    convert_resources_to_langchain_tool,
    list_mcp_resource_templates,
    list_mcp_resources,
)

@pytest.mark.asyncio
async def test_convert_resources_to_langchain_tool():
    mock_session = AsyncMock(spec=ClientSession)
    
    # Mock resources and templates
    mock_resource = Resource(
        uri="test://resource",
        name="Test Resource",
        description="Test Description",
        mimeType="text/plain",
        size=100,
        annotations={}
    )
    mock_template = ResourceTemplate(
        uriTemplate="test://template/{param}",
        name="Test Template",
        description="Test Template Description",
        mimeType="text/plain",
        annotations={}
    )
    
    # Setup mock responses
    mock_session.list_resources.return_value = ListResourcesResult(
        resources=[mock_resource]
    )
    mock_session.list_resource_templates.return_value = ListResourceTemplatesResult(
        resourceTemplates=[mock_template]
    )
    
    # Test tool creation and execution
    tool = await convert_resources_to_langchain_tool(mock_session)
    
    assert isinstance(tool, BaseTool)
    assert tool.name == "get_mcp_resource"
    assert "Test Resource" in tool.description
    assert "Test Template" in tool.description

@pytest.mark.asyncio
async def test_list_mcp_resources():
    mock_session = AsyncMock(spec=ClientSession)
    
    # Mock resources
    mock_resource = Resource(
        uri="test://resource",
        name="Test Resource",
        description="Test Description",
        mimeType="text/plain",
        size=100,
        annotations={}
    )
    
    # Setup mock response
    mock_session.list_resources.return_value = ListResourcesResult(
        resources=[mock_resource]
    )
    
    # Test resource listing
    resources = await list_mcp_resources(mock_session)
    
    assert resources == [mock_resource]

@pytest.mark.asyncio
async def test_list_mcp_resource_templates():
    mock_session = AsyncMock(spec=ClientSession)
    
    # Mock templates
    mock_template = ResourceTemplate(
        uriTemplate="test://template/{param}",
        name="Test Template",
        description="Test Template Description",
        mimeType="text/plain",
        annotations={}
    )
    
    # Setup mock response
    mock_session.list_resource_templates.return_value = ListResourceTemplatesResult(
        resourceTemplates=[mock_template]
    )
    
    # Test template listing
    templates = await list_mcp_resource_templates(mock_session)
    
    assert templates == [mock_template]