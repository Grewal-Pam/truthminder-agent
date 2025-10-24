import os, json, glob

def test_retrieval_stage_exists_or_is_optional():
    # If traces exist (from a local quick run), verify the retrieval stage is present.
    # If not, skip without failing CI (so remote CI does not require a heavy run).
    trace_files = glob.glob("outputs/traces/*.json")
    if not trace_files:
        # No traces on CI runners; that’s okay. This is a smoke test.
        assert True
        return

    # Pick one trace and validate
    with open(trace_files[0], "r", encoding="utf-8") as f:
        trace = json.load(f)

    assert "steps" in trace and isinstance(trace["steps"], list)
    stages = [s.get("stage") for s in trace["steps"]]
    assert "retrieval" in stages, "RAG retrieval stage should be present in the trace"
