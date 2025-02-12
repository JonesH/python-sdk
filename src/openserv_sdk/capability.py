from typing import Callable, Dict, Any, List, TypeVar, Optional, Generic
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class Capability(Generic[T]):
    """Represents a capability that an agent can perform."""
    
    def __init__(
        self,
        name: str,
        description: str,
        schema: T,
        run: Callable[[Dict[str, Any], List[Dict[str, str]]], str]
    ):
        self.name = name
        self.description = description
        self.schema = schema
        self.run = run
