"""阅读证据路径（trace_reading）算法测试。"""

import pytest

from app.assembler import AssemblyError
from app.tracer import TraceReadingError, trace_reading

SEGS3 = ["学而?习之", "时习之不", "之不亦乐乎"]  # 学而时习之不亦乐乎


def _ev(r):
    return [(e["segment"], e["pos"]) for e in r["evidence"]]


def test_full_interval_walks_segments_with_wildcard_kept_on_early_char():
    # 最早合法的 0->1 换段在“习”前（交界两侧“时”“习”均含段 0、1），
    # 故“时”只能取段 0 的 ?；路径 0,0,?,习 -> 段1 之 -> 段2 不亦乐乎。
    # “习”取段 0(p3) 或此时换段取段 1(p1) 都是 (2 换段,1 个 ?)，故有同分路径。
    r = trace_reading(SEGS3, 0, 9)
    assert r["fragment"] == "学而时习之不亦乐乎"
    assert (r["switch_count"], r["wildcard_count"]) == (2, 1)
    assert r["ambiguous"] is True
    assert _ev(r) == [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 2),
        (2, 1), (2, 2), (2, 3), (2, 4),
    ]
    assert r["wildcard_picks"] == [
        {"pos": 2, "segment": 0, "source_pos": 2}
    ]
    assert [s["pos"] for s in r["switches"]] == [4, 5]
    assert r["switches"][0] == {
        "pos": 4, "from_segment": 0, "from_pos": 3,
        "to_segment": 1, "to_pos": 2,
    }


def test_runs_give_merged_and_original_intervals():
    r = trace_reading(SEGS3, 0, 9)
    runs = r["runs"]
    assert [run["segment"] for run in runs] == [0, 1, 2]
    assert runs[0]["merged"] == {
        "start": 0, "end": 4, "fragment": "学而时习"
    }
    assert runs[0]["original"] == {
        "start": 0, "end": 4, "fragment": "学而?习"
    }
    assert runs[1]["merged"] == {"start": 4, "end": 5, "fragment": "之"}
    assert runs[1]["original"] == {"start": 2, "end": 3, "fragment": "之"}
    assert runs[2]["merged"] == {
        "start": 5, "end": 9, "fragment": "不亦乐乎"
    }
    assert runs[2]["original"] == {
        "start": 1, "end": 5, "fragment": "不亦乐乎"
    }


def test_subinterval_inside_overlap_zero_switches():
    # [3,6) 习之不：全部可由段 1 连续取字 pos 1,2,3
    r = trace_reading(SEGS3, 3, 6)
    assert r["fragment"] == "习之不"
    assert (r["switch_count"], r["wildcard_count"]) == (0, 0)
    assert _ev(r) == [(1, 1), (1, 2), (1, 3)]
    assert r["runs"] == [
        {
            "segment": 1,
            "merged": {"start": 3, "end": 6, "fragment": "习之不"},
            "original": {"start": 1, "end": 4, "fragment": "习之不"},
        }
    ]
    assert r["switches"] == []
    assert r["wildcard_picks"] == []


def test_empty_interval_is_valid_empty_path():
    r = trace_reading(SEGS3, 5, 5)
    assert r["fragment"] == ""
    assert r["evidence"] == []
    assert r["runs"] == []
    assert r["switches"] == []
    assert r["wildcard_picks"] == []
    assert (r["switch_count"], r["wildcard_count"]) == (0, 0)
    assert r["ambiguous"] is False


def test_empty_interval_at_end_of_text():
    r = trace_reading(SEGS3, 9, 9)
    assert r["start"] == 9 and r["end"] == 9
    assert r["fragment"] == ""


def test_switch_must_land_on_two_sided_overlap():
    # 段 0=甲乙丙、段 1=乙丙丁；甲仅段 0、丁仅段 1。
    # 换段只能落在乙|丙 交界（两侧都同时含段 0、1）；
    # 丙|丁 交界的丁不含段 0，故换段不能拖到最后。
    r = trace_reading(["甲乙丙", "乙丙丁"], 0, 4)
    assert r["fragment"] == "甲乙丙丁"
    assert (r["switch_count"], r["wildcard_count"]) == (1, 0)
    assert _ev(r) == [(0, 0), (0, 1), (1, 1), (1, 2)]
    assert r["switches"] == [
        {
            "pos": 2, "from_segment": 0, "from_pos": 1,
            "to_segment": 1, "to_pos": 1,
        }
    ]
    # 段 1 的 run 原文区间连续（乙? 不取——段 1 从丙 p1 起取）
    assert r["runs"][1]["original"] == {
        "start": 1, "end": 3, "fragment": "丙丁"
    }


def test_mismatch_source_cannot_evidenced():
    # 段 1 的 X 与输出字丙异字冲突：该来源不可作证
    r = trace_reading(["甲乙丙丁", "乙X丁"], 0, 4)
    assert r["fragment"] == "甲乙丙丁"
    assert _ev(r) == [(0, 0), (0, 1), (0, 2), (0, 3)]


def test_identical_segments_make_ambiguous_optimal_paths():
    r = trace_reading(["甲乙丙丁", "甲乙丙丁"], 0, 4)
    assert (r["switch_count"], r["wildcard_count"]) == (0, 0)
    assert r["ambiguous"] is True
    # 字典序取首：段序号 0
    assert _ev(r) == [(0, 0), (0, 1), (0, 2), (0, 3)]


def test_ambiguity_within_subinterval():
    r = trace_reading(["甲乙丙丁", "甲乙丙丁"], 1, 3)
    assert r["ambiguous"] is True
    assert _ev(r) == [(0, 1), (0, 2)]


def test_switch_count_dominates_wildcard_count():
    # 段 0 含 ?（乙位），段 1 同文：全取段 0 为 0 换段 1 个 ?；
    # 换段避开 ? 需 2 次换段（戊仅段 0）。换段次数优先，故选段 0 的 ?。
    r = trace_reading(["甲?丙丁戊", "甲乙丙丁"], 0, 5)
    assert r["fragment"] == "甲乙丙丁戊"
    assert (r["switch_count"], r["wildcard_count"]) == (0, 1)
    assert _ev(r) == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    assert r["wildcard_picks"] == [
        {"pos": 1, "segment": 0, "source_pos": 1}
    ]


def test_wildcard_minimized_when_switch_count_equal():
    # 区间 [1,4) 乙丙丁：段 0 的乙位是 ?。
    # 取段 1 连续 (乙,丙,丁)：0 换段、0 个 ?，优于取段 0（1 个 ?）。
    r = trace_reading(["甲?丙丁戊", "甲乙丙丁"], 1, 4)
    assert r["fragment"] == "乙丙丁"
    assert (r["switch_count"], r["wildcard_count"]) == (0, 0)
    assert _ev(r) == [(1, 1), (1, 2), (1, 3)]


def test_no_complete_path_when_switch_blocked_by_gap():
    # 拼接结果：甲乙丙丁戊己辛，其中
    #   丁(pos3) 仅段 0（对齐时段 1 此处空缺），
    #   戊(pos4) 段 0、1，己(pos5) 段 0(己)、1(庚)异字，辛(pos6) 仅段 1。
    # 区间 [3,7) 丁戊己辛：丁字必须取段 0，辛字必须取段 1，而唯一可能的
    # 0->1 换段交界在己|辛——辛不含段 0，换段非法；同段连续又无法跨越空缺，
    # 故不存在完整路径。
    segs = ["甲乙丙丁戊己", "乙丙戊庚辛"]
    with pytest.raises(TraceReadingError):
        trace_reading(segs, 3, 7)
    # 较短的相邻区间各自可行：[3,6) 全取段 0，[4,7) 同样无解（己|辛非法）
    assert _ev(trace_reading(segs, 3, 6)) == [
        (0, 3), (0, 4), (0, 5)
    ]
    with pytest.raises(TraceReadingError):
        trace_reading(segs, 4, 7)


def test_no_complete_path_when_continuity_broken():
    # 同上构造：异字冲突位（段 1 的庚 vs 输出己）不能作证，且无法以合法换段
    # 绕到段 1 覆盖辛；从丁（仅段 0）到辛（仅段 1）无任何完整路径。
    segs = ["甲乙丙丁戊己", "乙丙戊庚辛"]
    with pytest.raises(TraceReadingError):
        trace_reading(segs, 3, 7)


def test_interval_out_of_range_raises():
    with pytest.raises(TraceReadingError):
        trace_reading(SEGS3, 0, 10)
    with pytest.raises(TraceReadingError):
        trace_reading(SEGS3, 9, 10)


def test_negative_or_reversed_interval_raises():
    with pytest.raises(TraceReadingError):
        trace_reading(SEGS3, -1, 5)
    with pytest.raises(TraceReadingError):
        trace_reading(SEGS3, 5, 2)


def test_assembly_failure_propagates():
    with pytest.raises(AssemblyError):
        trace_reading(["孔子曰未知", "未知生焉", "完全无关文本"], 0, 1)


def test_response_reuses_full_assembly_text_and_sources():
    r = trace_reading(SEGS3, 2, 6)
    assert r["text"] == "学而时习之不亦乐乎"
    assert len(r["sources"]) == len(r["text"]) == 9
    assert r["fragment"] == "时习之不"


def test_evidence_original_chars_match_output_or_wildcard():
    from app.aligner import WILDCARD

    r = trace_reading(SEGS3, 0, 9)
    # 全部证据原字要么等于对应输出字、要么为 ?
    for ev, out_ch in zip(r["evidence"], r["fragment"]):
        orig = SEGS3[ev["segment"]][ev["pos"]]
        assert orig == out_ch or orig == WILDCARD


def test_same_segment_positions_strictly_increment_by_one():
    r = trace_reading(SEGS3, 0, 9)
    ev = _ev(r)
    for (s, q), (s2, q2) in zip(ev, ev[1:]):
        if s == s2:
            assert q2 == q + 1


def test_response_internal_consistency():
    r = trace_reading(SEGS3, 0, 9)
    n = len(r["fragment"])
    assert len(r["evidence"]) == n
    # runs 按序铺满整个区间、互不重叠
    runs = r["runs"]
    assert runs
    assert runs[0]["merged"]["start"] == r["start"]
    assert runs[-1]["merged"]["end"] == r["end"]
    for a, b in zip(runs, runs[1:]):
        assert a["merged"]["end"] == b["merged"]["start"]
    # 每个 run 内同段、原文区间连续，且片段确实取自该原始段
    for run in runs:
        seg = run["segment"]
        ms, me = run["merged"]["start"], run["merged"]["end"]
        os_, oe = run["original"]["start"], run["original"]["end"]
        assert me - ms == oe - os_
        assert run["original"]["fragment"] == SEGS3[seg][os_:oe]
        assert run["merged"]["fragment"] == r["text"][ms:me]
    # 计数与列表一致
    assert len(r["switches"]) == r["switch_count"]
    assert len(r["wildcard_picks"]) == r["wildcard_count"]
    # wildcard_picks 正是 evidence 中原字为 ? 的位置
    wp = {(w["pos"], w["segment"], w["source_pos"]) for w in r["wildcard_picks"]}
    expect_wp = set()
    for t, ev in enumerate(r["evidence"]):
        if SEGS3[ev["segment"]][ev["pos"]] == "?":
            expect_wp.add((r["start"] + t, ev["segment"], ev["pos"]))
    assert wp == expect_wp
    # 每次换段的两侧选择与 switches 一致
    for sw in r["switches"]:
        t = sw["pos"] - r["start"]
        assert r["evidence"][t - 1] == {
            "segment": sw["from_segment"], "pos": sw["from_pos"]
        }
        assert r["evidence"][t] == {
            "segment": sw["to_segment"], "pos": sw["to_pos"]
        }
