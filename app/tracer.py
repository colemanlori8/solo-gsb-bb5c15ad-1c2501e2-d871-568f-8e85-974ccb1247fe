"""阅读证据路径：为拼接稿区间逐字回溯“读到的原字”。

提交与 :func:`app.assembler.assemble` 相同限制的有序 segments 与拼接稿上的
左闭右开区间 ``[start, end)``，先复用其拼接得到拼接文本与逐字来源，再为区间
内**每个输出字**从其逐字来源中选取一个原始段原字作为证据：

- 仅当**原字等于输出字**、或**原字为 ``?``**（未辨认字）时，该来源才有资格
  作为该字的证据（异字冲突一方的原字不可作证）；
- 同一段内连续取字时，原文下标必须**逐次加一**（不允许跳字、倒读）；
- **换段**（相邻两字取自不同原始段）仅可发生在交界两侧的输出字**各自同时
  含有换前段与换后段来源**处——即交界左字 ``text[pos-1]`` 与右字
  ``text[pos]`` 的逐字来源中都同时列出这两个原始段（两侧重叠区各至少一字，
  换段才落在真实的续接重叠上）。

为整个区间求一条覆盖每个输出字的完整证据路径，依次**最小化换段次数**、
**最小化所选 ``?`` 数**；同分时按各字所选 **(段序号, 原文下标)** 序列字典序
取首（等价于段序号序列、再原文下标序列字典序），并以 ``ambiguous`` 标记是否
还存在其他同分路径。

动态规划：区间长度至多等于各段总长上限 800，每个输出字的候选证据至多来自
全部原始段（2～5 个），以前向 DP 逐状态保留字典序最优的 (换段数, ? 数) 与
取得它的路径数（只区分 1 条 / 多条），以后向 DP 记录每个状态到区间末尾的
最优后缀代价，前向贪心按 (段序号, 原文下标) 升序逐字回溯即得字典序最小的
最优路径。

区间非法（越界、起点大于终点）或不存在完整证据路径时抛出
:class:`TraceReadingError`，由接口层转为 422。本模块只做回溯，不改写拼接稿。
"""

from __future__ import annotations

from .aligner import WILDCARD
from .assembler import assemble


class TraceReadingError(ValueError):
    """区间非法，或不存在覆盖整个区间的完整证据路径。"""


def _eligible_candidates(
    segments: list[str],
    text: str,
    sources: list[list[dict]],
) -> list[dict[tuple[int, int], bool]]:
    """逐输出位置列出有资格作证的 ``(段序号, 原文下标) -> 原字是否为 ?``。

    同一位置上同段同下标的重复来源只计一次；仅保留原字等于输出字、或原字
    为 ``?`` 的来源（异字冲突一方被排除）。
    """
    layers: list[dict[tuple[int, int], bool]] = []
    for pos, srcs in enumerate(sources):
        out_ch = text[pos]
        cands: dict[tuple[int, int], bool] = {}
        for src in srcs:
            s, q = src["segment"], src["pos"]
            orig = segments[s][q]
            if orig == out_ch or orig == WILDCARD:
                cands.setdefault((s, q), orig == WILDCARD)
        layers.append(cands)
    return layers


def _can_switch(seg_sets: list[set[int]], t: int, s_from: int, s_to: int) -> bool:
    """换段是否合法：交界两侧各至少一字同时含换前段与换后段来源。

    交界位于区间内第 ``t-1`` 字与第 ``t`` 字（全局拼接位置 ``start+t-1`` 与
    ``start+t``）之间；两侧相邻字的**逐字来源**（不论该来源原字是否有资格
    作证）中须都同时列出 ``s_from``、``s_to`` 两个原始段。
    """
    left, right = seg_sets[t - 1], seg_sets[t]
    return s_from in left and s_to in left and s_from in right and s_to in right


def _forward_dp(
    cand_layers: list[dict[tuple[int, int], bool]],
    seg_sets: list[set[int]],
) -> list[dict[tuple[int, int], tuple[int, int, int]]]:
    """前向 DP：逐状态记录最优前缀代价与同分路径数。

    ``front[t][(s, q)] = (换段数, 所选 ? 数, ways)``，``ways`` 达到 2 即封顶。
    同状态只保留最小的 (换段数, ? 数)：未来后缀代价只取决于当前状态，故被
    支配的前缀不可能在全局最优中翻盘。

    每条合法转移只有两类：

    - **同段续读**：目标 ``(s, q)`` 的唯一同段前驱是 ``(s, q-1)``；
    - **换段**：代价只取决于换前段序号（某层每个段至多一个候选状态，合法
      入段集合对该层所有目标相同），故只需按换前段取最优前驱，与前驱原文
      下标无关。

    每层候选状态至多为段数（2～5 个），整体约 O(区间长度 × 段数²)。
    """
    n = len(cand_layers)
    front: list[dict[tuple[int, int], tuple[int, int, int]]] = []
    # 元组为 (换段数, 所选 ? 数, ways)：首字换段数恒为 0、? 数取该候选是否为 ?
    front.append(
        {
            key: (0, 1 if is_wildcard else 0, 1)
            for key, is_wildcard in cand_layers[0].items()
        }
    )
    for t in range(1, n):
        prev_layer = front[t - 1]
        cur_cands = cand_layers[t]

        # 按段汇总前驱状态：每段的最小 (换段数, ? 数)（不含 ? 增量，? 只与
        # 当前候选有关）及取得它的路径数；同时供换段转移枚举。
        best_by_seg: dict[int, tuple[int, int, int]] = {}
        for (s_from, _q_from), (sw, wc, ways_from) in prev_layer.items():
            old = best_by_seg.get(s_from)
            if old is None or (sw, wc) < (old[0], old[1]):
                best_by_seg[s_from] = (sw, wc, ways_from)
            elif (sw, wc) == (old[0], old[1]):
                best_by_seg[s_from] = (sw, wc, min(2, old[2] + ways_from))

        cur: dict[tuple[int, int], tuple[int, int, int]] = {}
        for (s_to, q_to), is_wildcard in cur_cands.items():
            wc_add = 1 if is_wildcard else 0
            best: tuple[int, int] | None = None
            ways_best = 0

            # 同段续读：唯一前驱 (s_to, q_to - 1)
            cont = prev_layer.get((s_to, q_to - 1))
            if cont is not None:
                best = (cont[0], cont[1] + wc_add)
                ways_best = cont[2]

            # 换段：枚举合法入段的换前段，取该段最优前驱
            for s_from, (sw, wc, ways_from) in best_by_seg.items():
                if s_from == s_to:
                    continue
                if not _can_switch(seg_sets, t, s_from, s_to):
                    continue
                cost = (sw + 1, wc + wc_add)
                if best is None or cost < best:
                    best, ways_best = cost, ways_from
                elif cost == best:
                    ways_best = min(2, ways_best + ways_from)

            if best is not None:
                cur[(s_to, q_to)] = (best[0], best[1], ways_best)
        front.append(cur)
    return front


def _suffix_costs(
    cand_layers: list[dict[tuple[int, int], bool]],
    seg_sets: list[set[int]],
) -> list[dict[tuple[int, int], tuple[int, int] | None]]:
    """后向 DP：``suf[t][key]`` 为该字选定 ``key`` 后到区间末尾的最优增量。

    增量计**第 ``t`` 字之后**的全部代价（交界 t→t+1 是否换段，以及
    第 ``t+1`` 字起各字的 ``?``；不含第 ``t`` 字自身）：最后一字任何候选的
    后缀增量均为 ``(0, 0)``；不存在可达末尾的合法后续时记为 ``None``。

    转移结构同 :func:`_forward_dp`：同段续读的唯一后继是 ``(s, q+1)``，
    换段的代价只取决于换后段（合法出段集合在本层对所有前驱相同），与双方
    原文下标无关——下一字取该段是否会引入 ``?`` 只取决于该层段内是否至少有
    一个已知字候选（有则增量 0，否则 1）。
    """
    n = len(cand_layers)
    suf: list[dict[tuple[int, int], tuple[int, int] | None]] = [
        dict.fromkeys(layer, (0, 0)) for layer in cand_layers
    ]
    for t in range(n - 2, -1, -1):
        next_suf = suf[t + 1]
        next_layer = cand_layers[t + 1]

        # 每个换后段：下一字落在此段时引入的最小 ? 增量（段内有已知字则为 0）
        wc_by_seg = {
            s: 0
            for (s, _q), is_wildcard in next_layer.items()
            if not is_wildcard
        }
        for (s, _q), is_wildcard in next_layer.items():
            wc_by_seg.setdefault(s, 1 if is_wildcard else 0)

        # 每个换后段在 ? 增量（由段决定、段内常数）之上的最小“再之后”后缀
        tail_by_seg: dict[int, tuple[int, int] | None] = {}
        for (s_next, _q_next), tail in next_suf.items():
            old = tail_by_seg.get(s_next)
            if tail is not None and (old is None or tail < old):
                tail_by_seg[s_next] = tail

        for (s_from, q_from) in cand_layers[t]:
            best: tuple[int, int] | None = None

            # 同段续读：唯一后继 (s_from, q_from + 1)
            cont = next_suf.get((s_from, q_from + 1))
            if cont is not None:
                best = (cont[0], wc_by_seg[s_from] + cont[1])

            # 换段：一次换段 + 下一字 ? 增量 + 该后继状态的再之后后缀
            for s_next, tail in tail_by_seg.items():
                if s_next == s_from or tail is None:
                    continue
                if not _can_switch(seg_sets, t + 1, s_from, s_next):
                    continue
                cost = (1 + tail[0], wc_by_seg[s_next] + tail[1])
                if best is None or cost < best:
                    best = cost
            suf[t][(s_from, q_from)] = best
    return suf


def trace_reading(segments: list[str], start: int, end: int) -> dict:
    """为拼接稿 ``[start, end)`` 求字典序最优的完整证据路径。

    返回 ``{"text", "sources", "start", "end", "fragment", "switch_count",
    "wildcard_count", "evidence", "runs", "switches", "wildcard_picks",
    "ambiguous"}``；拼接阶段失败原样抛出
    :class:`~app.assembler.AssemblyError`，区间非法或无完整路径抛出
    :class:`TraceReadingError`。
    """
    result = assemble(segments)
    text, sources = result["text"], result["sources"]
    total = len(text)

    if not isinstance(start, int) or not isinstance(end, int):
        raise TraceReadingError("区间起点、终点必须为整数")
    if start < 0 or end < 0 or start > end or end > total:
        raise TraceReadingError(
            f"非法区间 [{start}, {end})：拼接文本共 {total} 字，"
            "要求 0 ≤ start ≤ end ≤ 拼接文本长度"
        )

    window_sources = sources[start:end]
    cand_layers = _eligible_candidates(
        segments, text[start:end], window_sources
    )
    # 各输出字来源所含原始段集合（按区间内相对位置），用于换段合法性判定
    seg_sets = [{src["segment"] for src in srcs} for srcs in window_sources]
    length = end - start

    # 空区间：唯一的空证据路径，0 次换段、0 个 ?
    if length == 0:
        return {
            "text": text,
            "sources": sources,
            "start": start,
            "end": end,
            "fragment": "",
            "switch_count": 0,
            "wildcard_count": 0,
            "evidence": [],
            "runs": [],
            "switches": [],
            "wildcard_picks": [],
            "ambiguous": False,
        }

    front = _forward_dp(cand_layers, seg_sets)

    best: tuple[int, int] | None = None
    ways_total = 0
    for sw, wc, ways in front[-1].values():
        cost = (sw, wc)
        if best is None or cost < best:
            best, ways_total = cost, ways
        elif cost == best:
            ways_total = min(2, ways_total + ways)
    if best is None:
        raise TraceReadingError(
            f"区间 [{start}, {end})（{length} 字）不存在覆盖全部输出字的完整"
            "证据路径：某字无“原字等于输出字或为 ?”的来源，或同段连续取字、"
            "换段重叠要求无法同时满足"
        )

    suf = _suffix_costs(cand_layers, seg_sets)

    # 前向贪心：每一步在仍能取得全局最优的候选中按 (段序号, 原文下标) 取最小
    evidence: list[tuple[int, int]] = []
    prefix_cost = (0, 0)
    prev_key: tuple[int, int] | None = None
    for t in range(length):
        chosen = None
        for key in sorted(cand_layers[t]):
            s_to, q_to = key
            if t == 0:
                added = (0, 1 if cand_layers[0][key] else 0)
            else:
                assert prev_key is not None
                s_from, q_from = prev_key
                if s_from == s_to:
                    if q_to != q_from + 1:
                        continue
                    added = (0, 1 if cand_layers[t][key] else 0)
                elif not _can_switch(seg_sets, t, s_from, s_to):
                    continue
                else:
                    added = (1, 1 if cand_layers[t][key] else 0)
            tail = suf[t].get(key)
            if tail is None:
                continue
            total_cost = (
                prefix_cost[0] + added[0] + tail[0],
                prefix_cost[1] + added[1] + tail[1],
            )
            if total_cost == best:
                chosen = key
                prefix_cost = (prefix_cost[0] + added[0], prefix_cost[1] + added[1])
                break
        if chosen is None:  # pragma: no cover - 前向 DP 已保证存在完整路径
            raise TraceReadingError(
                f"区间 [{start}, {end}) 不存在覆盖全部输出字的完整证据路径"
            )
        evidence.append(chosen)
        prev_key = chosen

    switch_count, wildcard_count = best
    ambiguous = ways_total > 1

    # 连续同段选择合并为每段路径 run
    runs: list[dict] = []
    switches: list[dict] = []
    wildcard_picks: list[dict] = []
    run_start = 0
    for t, key in enumerate(evidence):
        s, q = key
        if cand_layers[t][key]:
            wildcard_picks.append(
                {"pos": start + t, "segment": s, "source_pos": q}
            )
        if t > 0:
            s_prev, q_prev = evidence[t - 1]
            if s != s_prev:
                switches.append(
                    {
                        "pos": start + t,
                        "from_segment": s_prev,
                        "from_pos": q_prev,
                        "to_segment": s,
                        "to_pos": q,
                    }
                )
                _emit_run(runs, segments, text, start, evidence, run_start, t)
                run_start = t
    _emit_run(runs, segments, text, start, evidence, run_start, length)

    return {
        "text": text,
        "sources": sources,
        "start": start,
        "end": end,
        "fragment": text[start:end],
        "switch_count": switch_count,
        "wildcard_count": wildcard_count,
        "evidence": [
            {"segment": s, "pos": q} for s, q in evidence
        ],
        "runs": runs,
        "switches": switches,
        "wildcard_picks": wildcard_picks,
        "ambiguous": ambiguous,
    }


def _emit_run(
    runs: list[dict],
    segments: list[str],
    text: str,
    start: int,
    evidence: list[tuple[int, int]],
    run_start: int,
    run_end: int,
) -> None:
    """把 ``evidence[run_start:run_end]``（同段连续选择）登记为一段路径。"""
    seg = evidence[run_start][0]
    q_first = evidence[run_start][1]
    q_last = evidence[run_end - 1][1]
    g_start, g_end = start + run_start, start + run_end
    runs.append(
        {
            "segment": seg,
            "merged": {
                "start": g_start,
                "end": g_end,
                "fragment": text[g_start:g_end],
            },
            "original": {
                "start": q_first,
                "end": q_last + 1,
                "fragment": segments[seg][q_first:q_last + 1],
            },
        }
    )
