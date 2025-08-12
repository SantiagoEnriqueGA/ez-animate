import os
import shutil
import subprocess
from pathlib import Path

def convert_gifs_to_thumbnails(folder_path, output_folder=None, scale_percent=50):
    """Convert all .gif files in the specified folder to thumbnails.

    Parameters:
    - folder_path (str): Path to the folder containing .gif files.
    - output_folder (str, optional): Where to save thumbnails. Defaults to input folder.
    - scale_percent (int): Scale of the thumbnail as a percentage of the original size.
    """
    folder = Path(folder_path)
    gif_files = list(folder.glob("*.gif"))

    if not gif_files:
        print("No .gif files found.")
        return

    if output_folder:
        output_dir = Path(output_folder)
        if output_dir.exists():
            print(f"Removing existing output folder: {output_dir}")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = folder

    for gif_path in gif_files:
        thumbnail_name = gif_path.stem + "_thumbnail.png"
        thumbnail_path = output_dir / thumbnail_name

        scale_expr = f"scale=iw*{scale_percent}/100:ih*{scale_percent}/100"

        cmd = [
            "ffmpeg",
            "-i", str(gif_path),
            "-vframes", "1",
            "-vf", scale_expr,
            str(thumbnail_path)
        ]

        try:
            print(f"Generating thumbnail: {thumbnail_path} ({scale_percent}%)")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Failed to generate thumbnail for {gif_path.name}: {e}")

if __name__ == "__main__":
    convert_gifs_to_thumbnails("docs/plots/gallery", output_folder="docs/plots/thumbnails", scale_percent=100)
