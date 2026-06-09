import pytest

from evaluation.registry import Registry


def test_registry_builds_registered_item():
    registry = Registry("task")
    registry.register("fake", lambda cfg: {"cfg": cfg})

    built = registry.build({"name": "fake", "x": 1})

    assert built == {"cfg": {"name": "fake", "x": 1}}


def test_registry_reports_available_names():
    registry = Registry("head")
    registry.register("linear", lambda cfg: cfg)

    with pytest.raises(ValueError, match="unknown head 'attn'.*linear"):
        registry.build({"name": "attn"})


def test_registry_requires_name():
    registry = Registry("task")

    with pytest.raises(ValueError, match="requires a 'name'"):
        registry.build({})
