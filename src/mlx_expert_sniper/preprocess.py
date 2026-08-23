"""
mlx-sniper preprocess — split an MLX 4-bit MoE checkpoint into sniper
streaming format (pinned.safetensors + bin/layer_XX.bin).

Thin wrapper over download._preprocess for models you have already
downloaded (e.g. via huggingface-cli) instead of `mlx-sniper download`.

NOTE: source shards are NOT deleted by this path (unlike download,
which deletes as it goes to save disk).
"""
import argparse
import glob
import os


def preprocess(src_dir, out_dir):
    src_dir = os.path.expanduser(src_dir)
    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(os.path.join(src_dir, "config.json")):
        raise SystemExit(f"No config.json in {src_dir} — is this an MLX model dir?")
    if not glob.glob(os.path.join(src_dir, "model*.safetensors")):
        raise SystemExit(f"No model*.safetensors shards in {src_dir}")

    from .download import _preprocess
    _preprocess(src_dir, out_dir, delete_shards=False)
    print(f"\nDone. Next: mlx-sniper calibrate {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split an MLX 4-bit MoE model "
                                     "into sniper streaming format")
    parser.add_argument("src_dir", help="Downloaded MLX model directory")
    parser.add_argument("out_dir", help="Output streaming-format directory")
    args = parser.parse_args()
    preprocess(args.src_dir, args.out_dir)


if __name__ == "__main__":
    main()
