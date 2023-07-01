import subprocess


def capture(command):
    proc = subprocess.Popen(command,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    out,err = proc.communicate()
    return out, err, proc.returncode


def test_help():
    command = ["python", "-m", "pipeGEM.cli", "-h"]
    out, err, exitcode = capture(command)
    print(out, err, exitcode)