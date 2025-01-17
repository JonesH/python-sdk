from typing import TypeVar, Protocol, Dict, Any, List, Awaitable, Union, Generic
from pydantic import BaseModel
import inspect
from .types import AgentAction, ChatMessage

T = TypeVar('T', bound=BaseModel)

class CapabilityFunction(Protocol[T]):
    """Protocol defining the expected signature of a capability's run function."""
    def __call__(
        self,
        params: Dict[str, Union[T, AgentAction]],
        messages: List[Dict[str, str]]
    ) -> Union[str, Awaitable[str]]: ...

class Capability(Generic[T]):
    """
    A capability that can be added to an agent.
    
    Attributes:
        name: The unique name of the capability
        description: A description of what the capability does
        schema: The Pydantic model class defining the capability's parameters
        run: The function that implements the capability's behavior
    """
    def __init__(
        self,
        name: str,
        description: str,
        schema: type[T],
        run: CapabilityFunction[T]
    ) -> None:
        """
        Initialize a new Capability instance.
        
        Args:
            name: The name of the capability
            description: A description of what the capability does
            schema: The Pydantic model class defining the capability's parameters
            run: The function that implements the capability's behavior
            
        Raises:
            TypeError: If schema is not a Pydantic model class
            ValueError: If run is not callable
        """
        if not issubclass(schema, BaseModel):
            raise TypeError("schema must be a Pydantic model class")
        if not callable(run):
            raise ValueError("run must be a callable")
            
        self.name = name
        self.description = description
        self.schema = schema
        
        # Ensure run is an async function
        if inspect.iscoroutinefunction(run):
            self.run = run
        else:
            # Convert sync function to async
            async def async_run(*args, **kwargs) -> str:
                return run(*args, **kwargs)
            self.run = async_run
