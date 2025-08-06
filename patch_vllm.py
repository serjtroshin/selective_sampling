#!/usr/bin/env python3
import os
import sys
import site
import shutil


def main():
    # Path to your custom model file
    custom_file = os.path.join("selective_sampling", "myllama.py")
    if not os.path.isfile(custom_file):
        print(f"Error: {custom_file} not found.")
        sys.exit(1)

    # Locate site-packages directory where vllm is installed
    site_packages = site.getsitepackages()
    vllm_models_path = None
    registry_path = None
    for sp in site_packages:
        candidate_models = os.path.join(sp, "vllm", "model_executor", "models")
        candidate_registry = os.path.join(candidate_models, "registry.py")
        if os.path.isdir(candidate_models) and os.path.isfile(candidate_registry):
            vllm_models_path = candidate_models
            registry_path = candidate_registry
            break

    if not vllm_models_path or not registry_path:
        print(
            "Could not locate vLLM models directory or registry.py. Is vllm installed?"
        )
        sys.exit(1)

    print(f"Located vLLM models directory: {vllm_models_path}")
    print(f"Located registry.py at: {registry_path}")

    # Copy the custom file
    dest_file = os.path.join(vllm_models_path, "myllama.py")
    shutil.copy2(custom_file, dest_file)
    print(f"Copied {custom_file} → {dest_file}")

    # Read registry.py content
    with open(registry_path, "r") as f:
        lines = f.readlines()

    new_entry = '    "MyCustomLlama": ("myllama", "MyCustomLlama"),\n'

    # Check if already patched
    if any("MyCustomLlama" in line for line in lines):
        print("MyCustomLlama is already registered. No changes made.")
        return

    # Insert new entry into _TEXT_GENERATION_MODELS dict
    patched = False
    with open(registry_path, "w") as f:
        for line in lines:
            f.write(line)
            if line.strip().startswith("_TEXT_GENERATION_MODELS = {") and not patched:
                f.write(new_entry)
                patched = True

    if patched:
        print("✅ Successfully patched registry.py with MyCustomLlama.")
    else:
        print("⚠️ Could not find _TEXT_GENERATION_MODELS dict in registry.py.")


if __name__ == "__main__":
    main()
