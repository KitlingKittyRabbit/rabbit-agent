"""Web 渲染纯逻辑测试：桥接 node 内置 test runner（零第三方依赖、零网络）。

node 缺失时跳过，保证默认套件在无 node 环境仍可跑。
"""

import shutil
import subprocess
from pathlib import Path

import pytest

TEST_FILE = Path(__file__).resolve().parents[1] / "web" / "timeline_logic.test.mjs"


def test_timeline_logic_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node 不可用，跳过 JS 逻辑测试")
    proc = subprocess.run(
        [node, "--test", str(TEST_FILE)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"node --test 失败:\n{proc.stdout}\n{proc.stderr}"
