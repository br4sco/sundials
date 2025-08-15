import pytest
from hypothesis import given, strategies as st

from dataclasses import dataclass
from typing import Any, Union, Callable, TypeVar, List, Tuple, Optional

from pydantic import BaseModel, validate_call
from pydantic_core import core_schema
