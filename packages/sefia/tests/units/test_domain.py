import glyff
import pytest

from sefia import Domain


def test_domain_requires_explicit_non_empty_execution_names() -> None:
    domain = Domain(glyff.Domain("tests.domain", version="1"))

    with pytest.raises(ValueError, match="name"):
        domain.infer(name="")
    with pytest.raises(ValueError, match="name"):
        domain.engrave(name="")
