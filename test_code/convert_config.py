import os

from config import config

os.chdir(os.path.dirname(os.path.abspath(__file__)))

cmd = "python " + config["program"]
for item in config["args"]:
    cmd = cmd + " " + item

print(cmd)
with open("test_code.sh", "a") as file:
    file.write(cmd + "\n")
