"""
Capability class for defining agent tools.
"""

from typing import TypeVar, Generic, Callable, Dict, Any, List, Awaitable, Union
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class Capability(Generic[T]):
    """
    Represents a tool/capability that an agent can use.
    
    Each capability has:
    - A unique name
    - A description of what it does
    - A Pydantic schema defining its parameters
    - A run function that executes the capability
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        schema: type[T],
        run: Callable[[Dict[str, Any], List[Dict[str, str]]], Union[str, Awaitable[str]]]
    ) -> None:
        """
        Initialize a new capability.
        
        Args:
            name: Unique name for the capability
            description: Description of what the capability does
            schema: Pydantic model class defining the parameters
            run: Function that implements the capability's behavior
                 Takes a dict with 'args' and optional 'action', plus a list of messages
                 Returns a string result or a coroutine that resolves to a string
        """
        self.name = name
        self.description = description
        self.schema = schema
        self._run = run
        
        # Validate schema
        if not issubclass(schema, BaseModel):
            raise ValueError("Schema must be a Pydantic model class")
            
        # Validate name
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Name must be a non-empty string")
            
        # Validate description
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Description must be a non-empty string")
            
        # Validate run function
        if not callable(run):
            raise ValueError("Run must be a callable")
            
    async def run(self, params: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
        """
        Execute the capability.
        
        Args:
            params: Dictionary containing:
                - args: Arguments matching the schema
                - action: Optional action context
            messages: List of chat messages for context
            
        Returns:
            String result from executing the capability
            
        Raises:
            ValidationError: If args don't match schema
            Exception: If execution fails
        """
        # Validate args against schema
        args = params.get('args', {})
        validated_args = self.schema.model_validate(args)
        
        # Call the run function
        result = self._run({"args": validated_args.model_dump(), "action": params.get("action")}, messages)
        
        # Handle both sync and async run functions
        if isinstance(result, Awaitable):
            result = await result
            
        return str(result)
        
    def __repr__(self) -> str:
        """String representation of the capability."""
        return f"Capability(name='{self.name}', description='{self.description}')"
