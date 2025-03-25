from typing import Any, Callable, Dict, List, Type, Union, TypeVar, Generic, Protocol, Awaitable, Optional
from pydantic import BaseModel
from .schema_types import ChatCompletionMessageParam

T = TypeVar('T', bound=BaseModel)

class CapabilityFuncParams(Generic[T]):
    """Parameters for a capability's run function."""
    args: T
    action: Optional[Dict[str, Any]] = None

class Capability(Generic[T]):
    """
    Represents a capability that an agent can perform.
    
    A capability has a name, description, schema for parameters validation,
    and a run function that implements its behavior.
    
    Type Parameters:
        T: The type of the parameters schema, must be a subclass of BaseModel
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        schema: Type[T],
        run: Callable[['Agent', CapabilityFuncParams[T], List[ChatCompletionMessageParam]], Union[str, Awaitable[str]]]
    ):
        """
        Initialize a new capability.
        
        Args:
            name: Unique name for the capability
            description: Description of what the capability does
            schema: Pydantic model defining the capability's parameters
            run: Function that implements the capability's behavior
                Takes (agent, params, messages) as arguments
                Returns a string or an awaitable that resolves to a string
        """
        self._name = name
        self._description = description
        self._schema = schema
        self._run = run
    
    @property
    def name(self) -> str:
        """The name of the capability."""
        return self._name
    
    @property
    def description(self) -> str:
        """A description of what the capability does."""
        return self._description
    
    @property
    def schema(self) -> Type[T]:
        """The Pydantic model that defines the parameters schema."""
        return self._schema
    
    @property
    def run(self) -> Callable[['Agent', CapabilityFuncParams[T], List[ChatCompletionMessageParam]], Union[str, Awaitable[str]]]:
        """The function that implements the capability's behavior."""
        return self._run