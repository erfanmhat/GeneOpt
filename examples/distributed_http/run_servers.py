"""Start N objective servers on consecutive ports: python run_servers.py N [BASE_PORT]"""

import subprocess
import sys
from pathlib import Path

n = int(sys.argv[1])
base = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
here = Path(__file__).resolve().parent
procs = [subprocess.Popen([sys.executable, "server.py", str(base + i)], cwd=here) for i in range(n)]
codes = []
try:
    codes = [p.wait() for p in procs]
finally:  # Ctrl-C or one server exiting stops the whole group: no orphans holding ports
    for p in procs:
        p.terminate()
sys.exit(max(codes, default=0))
