from __future__ import annotations

import inspect_reference_usd_candidates as inspection


def test_stage_traversal_includes_instance_proxies() -> None:
    marker = object()
    stage = object()

    class FakePrimRange:
        @staticmethod
        def Stage(actual_stage: object, predicate: object) -> tuple[object, object]:
            return actual_stage, predicate

    class FakeUsd:
        PrimRange = FakePrimRange

        @staticmethod
        def TraverseInstanceProxies() -> object:
            return marker

    assert inspection._traverse_stage_prims(FakeUsd, stage) == (stage, marker)
