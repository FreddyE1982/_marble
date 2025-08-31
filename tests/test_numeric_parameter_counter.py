import subprocess
import sys


def test_numeric_parameter_counts():
    proc = subprocess.run(
        [sys.executable, 'count_numeric_parameters.py'],
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout
    assert 'Total numeric parameters: 97' in out
    assert 'Learnable numeric parameters: 84' in out
    assert 'Non-learnable numeric parameters: 13' in out
