"""Apply the public client contract to the core test client."""

from sefia.llm import LLMCompletion
from sefia.testing import LLMClientCase, LLMClientContract, MockLLMClient
from typing_extensions import override


class TestMockLLMClientContract(LLMClientContract):
    @override
    def make_llm_client_case(self) -> LLMClientCase:
        completion = LLMCompletion(content="done")
        return LLMClientCase(MockLLMClient([completion]), completion)
