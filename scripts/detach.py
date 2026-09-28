import os, sys
if os.fork():          # parent returns immediately
    sys.exit(0)
os.setsid()            # child: new session, no longer tied to the tool's process group
os.execvp(sys.argv[1], sys.argv[1:])
