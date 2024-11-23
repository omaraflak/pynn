import subprocess

tensor_h = open("tensor.h", "r").read()
tensor_cu = open("tensor.cu", "r").read()
cpu_h = open("cpu.h", "r").read()
cpu_cu = open("cpu.cu", "r").read()
gpu_h = open("gpu.h", "r").read()
gpu_cu = open("gpu.cu", "r").read()
makefile = open("Makefile", "r").read()
tensor_py = open("tensor.py", "r").read()

code = f"""
tensor_h = \"\"\"
{tensor_h}
\"\"\"

tensor_cu = \"\"\"
{tensor_cu}
\"\"\"

cpu_h = \"\"\"
{cpu_h}
\"\"\"

cpu_cu = \"\"\"
{cpu_cu}
\"\"\"

gpu_h = \"\"\"
{gpu_h}
\"\"\"

gpu_cu = \"\"\"
{gpu_cu}
\"\"\"

makefile = \"\"\"
{makefile}
\"\"\"

tensor_py = \"\"\"
{tensor_py}
\"\"\"

open("tensor.h", "w").write(tensor_h)
open("tensor.cu", "w").write(tensor_cu)
open("cpu.h", "w").write(cpu_h)
open("cpu.cu", "w").write(cpu_cu)
open("gpu.h", "w").write(gpu_h)
open("gpu.cu", "w").write(gpu_cu)
open("Makefile", "w").write(makefile)
open("tensor.py", "w").write(tensor_py)
"""

subprocess.run("pbcopy", text=True, input=code)
print("copied!")
