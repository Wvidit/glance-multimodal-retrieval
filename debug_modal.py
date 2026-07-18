import modal
import os
from pathlib import Path

app = modal.App("debug")
volume = modal.Volume.from_name("glance-data")

@app.function(volumes={"/data": volume})
def check():
    path = Path("/data/images")
    if not path.exists():
        print("NO IMAGES DIR")
        return
    
    files = list(path.glob("*.jpg"))
    print(f"Found {len(files)} jpgs")
    for f in files[:5]:
        size = f.stat().st_size
        print(f"{f}: {size} bytes")
        # Try to read head
        with open(f, "rb") as fp:
            head = fp.read(10)
            print(f"Head: {head}")

if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            check.remote()
