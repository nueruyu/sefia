"""Apply the public client contract to the core test client."""

import pytest

from sefia.llm import LLMCompletion
from sefia.testing import LLMClientCase, LLMClientContract, MockLLMClient


class TestMockLLMClientContract(LLMClientContract):
    @pytest.fixture
    def llm_client_case(self) -> LLMClientCase:
        completion = LLMCompletion(content="done")
        return LLMClientCase(MockLLMClient([completion]), completion)
