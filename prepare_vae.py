"""Download the VAE only when explicitly preparing the optional local backend."""
from pathlib import Path
import argparse
import urllib.request

ROOT = Path(__file__).resolve().parent / "checkpoints" / "sd-vae-ft-mse"
BASE = "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/"


def main():
    parser = argparse.ArgumentParser(description="Prepare optional local VAE models")
    parser.add_argument("--tiny", action="store_true", help="Download the optional fast TAESD model (about 10 MB)")
    args = parser.parse_args()
    root = ROOT.parent / "taesd" if args.tiny else ROOT
    base = "https://huggingface.co/madebyollin/taesd/resolve/614f76814bbe30edbe2e627ace1c2234c81a2c0e/" if args.tiny else BASE
    root.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "diffusion_pytorch_model.safetensors"):
        dest = root / name
        if dest.is_file() and dest.stat().st_size > 100:
            print(f"Already present: {name}", flush=True)
            continue
        partial = dest.with_suffix(dest.suffix + ".part")
        try:
            print(f"Downloading {name}", flush=True)
            with urllib.request.urlopen(base + name, timeout=60) as response, partial.open("wb") as output:
                expected = int(response.headers.get("Content-Length", 0))
                total = reported = 0
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    total += len(chunk)
                    if total - reported >= 32 * 1024 * 1024:
                        print(f"{total / 2**20:.0f} MiB", flush=True)
                        reported = total
            if expected and partial.stat().st_size != expected:
                raise RuntimeError("Incomplete model download")
            partial.replace(dest)
        finally:
            partial.unlink(missing_ok=True)
    print(f"Model ready: {root}", flush=True)


if __name__ == "__main__":
    main()
