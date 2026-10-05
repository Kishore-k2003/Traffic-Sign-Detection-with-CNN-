import subprocess, sys
packages = ['numpy','matplotlib','seaborn','opencv-python','scikit-learn','tensorflow']
subprocess.check_call([sys.executable, '-m', 'pip', 'install', *packages])
