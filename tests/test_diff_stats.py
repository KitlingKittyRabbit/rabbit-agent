"""文件改动行数统计的纯函数测试。"""

from agent.core.diff_stats import accrue, count_lines, diff_payload, file_change


def test_count_lines() -> None:
    assert count_lines("") == 0
    assert count_lines("a") == 1
    assert count_lines("a\n") == 1
    assert count_lines("a\nb") == 2
    assert count_lines("a\nb\n") == 2


def test_file_change_write_and_edit() -> None:
    assert file_change("write_file", {"path": "a.txt", "content": "1\n2\n3\n"}) == \
        {"path": "a.txt", "added": 3, "removed": 0}
    assert file_change("edit_file", {"path": "a.txt", "old": "1\n2\n", "new": "1\nX\nY\n"}) == \
        {"path": "a.txt", "added": 3, "removed": 2}
    assert file_change("read_file", {"path": "a.txt"}) is None
    assert file_change("write_file", {}) is None


def test_accrue_and_payload_sorted() -> None:
    diff: dict = {}
    accrue(diff, {"path": "b.txt", "added": 2, "removed": 0})
    accrue(diff, {"path": "a.txt", "added": 1, "removed": 3})
    accrue(diff, {"path": "b.txt", "added": 1, "removed": 1})
    payload = diff_payload(diff)
    assert payload == {"files": [
        {"path": "a.txt", "added": 1, "removed": 3},
        {"path": "b.txt", "added": 3, "removed": 1},
    ]}
    assert diff_payload({}) is None
