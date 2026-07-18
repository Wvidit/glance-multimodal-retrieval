import modal
import os
from pathlib import Path

image = modal.Image.debian_slim(python_version="3.11").pip_install("Pillow")
app = modal.App("debug-pillow")
volume = modal.Volume.from_name("glance-data")

@app.function(image=image, volumes={"/data": volume})
def check():
    from PIL import Image
    path = Path("/data/images")
    files = list(path.glob("*.jpg"))
    for f in files[:5]:
        try:
            img = Image.open(f)
            img.convert("RGB")
            print(f"Successfully loaded {f}")
        except Exception as e:
            print(f"Error loading {f}: {repr(e)}")

if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            check.remote()
