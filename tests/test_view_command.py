import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('probe_status, expected', [(0, 'host'), (1, 'docker')])
def test_view_uses_host_only_when_dependency_probe_succeeds(tmp_path, probe_status, expected):
    root = Path(__file__).resolve().parents[1]
    shutil.copy(root / 'accord', tmp_path / 'accord')
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'bin').mkdir()
    python = tmp_path / 'bin/python3'
    python.write_text('#!/bin/bash\nif [ "$1" = "-c" ]; then exit "$PROBE_STATUS"; fi\n'
                      'echo "host $*" > "$RESULT"\n')
    python.chmod(0o755)
    docker = tmp_path / 'scripts/docker.sh'
    docker.write_text('#!/bin/bash\necho "docker $*" > "$RESULT"\n')
    docker.chmod(0o755)
    result = tmp_path / 'result'
    env = dict(os.environ, PATH=f'{tmp_path / "bin"}:{os.environ["PATH"]}',
               PROBE_STATUS=str(probe_status), RESULT=str(result))
    subprocess.run([str(tmp_path / 'accord'), 'view', 'runs/example'], env=env, check=True)
    output = result.read_text()
    assert output.startswith(expected + ' ')
    assert 'scripts/view_run.py --run runs/example' in output
