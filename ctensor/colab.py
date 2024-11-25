import subprocess

files = [
    "tensor.h",
    "tensor.cu",
    "cpu.h",
    "cpu.cu",
    "gpu.h",
    "gpu.cu",
    "Makefile",
    "tensor.py",
    "test_tensor.py",
]

names = [file.replace(".", "_") for file in files]
contents = [open(file, "r").read() for file in files]
code = "\n".join(
    f"{name} = \"\"\"\n{content}\n\"\"\""
    for name, content in zip(names, contents)
)
writes = "\n".join(
    f"open('{file}', 'w').write({name})"
    for file, name in zip(files, names)
)
all = code + "\n" + writes
all = all.replace("\\n", "\\\\n")

subprocess.run("pbcopy", text=True, input=all)
print("copied!")
