"""Check the experiment API and Claude availability before research."""
import subprocess

from autoarena import Benchmark

subprocess.run(['claude', '--version'], check=True)
