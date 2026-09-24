"""Verify native controller, bridge and provider dependencies."""
import subprocess
from arbor.coordinator.main import cli
from arbor.arena_bridge import recorded_branch
from arbor.core.llm.litellm_provider import LiteLLMProvider
from autoarena import Benchmark
import boto3

subprocess.run(["bwrap", "--die-with-parent", "--bind", "/", "/",
                "--dev", "/dev", "--proc", "/proc", "--", "true"], check=True)
print("Native Arbor controller and benchmark bridge are installed")
