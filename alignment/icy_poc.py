#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import glob
import os

java_path = "/usr/lib/jvm/java-8-openjdk-amd64/bin/java"
icy_path = "/home/ula/Downloads/icy_2.5.3/Icy/"  

icy_jar_path = f"{icy_path}icy.jar"
log4j_path = f"{icy_path}lib/log4j-1.2.17.jar"
protocols_plugin_jar = f"{icy_path}/plugins/adufour/protocols/Protocols.jar"

plugin_jars = []#glob.glob(os.path.join(icy_path, "plugins/**/*.jar"), recursive=True)


classpath = ":".join([log4j_path, icy_jar_path] + plugin_jars)

icy_executable = "icy.main.Icy"

protocol_plugin = "plugins.adufour.protocols.Protocols"
protocol_path = "/home/ula/twophoton/twophoton/icy_protocols/post_alignment_segmentation.protocol"


cmd = [
    java_path,
    "-cp", classpath,
    icy_executable,
    "-hl",
    "-x", protocol_plugin,
    f"protocol={protocol_path}"
]

cwd = "/home/ula/Downloads/icy-2.3.0.0-all/"

result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)

