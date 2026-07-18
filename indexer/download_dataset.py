"""
Downloads and curates a subset of the Fashionpedia dataset.
"""

import argparse
import json
import logging
import os
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ANNOTATIONS_FILE,
    DATA_DIR,
    FASHIONPEDIA_ANNOTATIONS_URL,
    FASHIONPEDIA_TRAIN_IMAGES_URL,
    IMAGES_DIR,
    TARGET_SUBSET_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_with_progress(url: str, dest_path: Path) -> None:
    """Download a file with a tqdm progress bar."""
    if dest_path.exists():
        logger.info(f"File already exists: {dest_path}")
        return

    logger.info(f"Downloading {url} to {dest_path}")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urlretrieve(url, filename=dest_path, reporthook=t.update_to)
    logger.info("Download complete.")


def parse_annotations(annotations_path: Path) -> dict:
    """Parse the Fashionpedia annotations JSON."""
    logger.info("Parsing annotations JSON...")
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    # Map IDs to names
    categories = {cat['id']: cat['name'].lower() for cat in data['categories']}
    attributes = {attr['id']: attr['name'].lower() for attr in data['attributes']}
    
    # Map image_id to image info
    images_info = {img['id']: img for img in data['images']}
    
    # Aggregate annotations per image
    image_annotations = defaultdict(lambda: {"categories": set(), "attributes": set()})
    
    for ann in tqdm(data['annotations'], desc="Processing annotations"):
        img_id = ann['image_id']
        cat_id = ann['category_id']
        attr_ids = ann.get('attribute_ids', [])
        
        if cat_id in categories:
            image_annotations[img_id]["categories"].add(categories[cat_id])
        
        for attr_id in attr_ids:
            if attr_id in attributes:
                image_annotations[img_id]["attributes"].add(attributes[attr_id])
                
    # Build final metadata structure
    metadata = []
    for img_id, info in images_info.items():
        if img_id not in image_annotations:
            continue
            
        cats = list(image_annotations[img_id]["categories"])
        attrs = list(image_annotations[img_id]["attributes"])
        
        # Skip images with very few annotations to ensure quality
        if not cats:
            continue
            
        # Create a structured caption
        cats_str = " and ".join(cats) if len(cats) > 0 else "clothing"
        attrs_str = ", ".join(attrs) if len(attrs) > 0 else ""
        structured_caption = f"A person wearing {cats_str}."
        if attrs_str:
            structured_caption = f"A person wearing {cats_str} with {attrs_str}."
            
        metadata.append({
            "image_id": img_id,
            "file_name": info["file_name"],
            "width": info["width"],
            "height": info["height"],
            "categories": cats,
            "attributes": attrs,
            "structured_caption": structured_caption,
            "annotation_count": len(cats) + len(attrs)
        })
        
    return metadata


def curate_subset(metadata: list, subset_size: int) -> list:
    """Select a diverse subset ensuring all categories are represented."""
    logger.info(f"Curating subset of {subset_size} images from {len(metadata)} total.")
    
    # Group by primary category
    by_category = defaultdict(list)
    for item in metadata:
        if item["categories"]:
            # Just group by the first category for balancing
            by_category[item["categories"][0]].append(item)
            
    # Sort each category group by annotation count (richer metadata first)
    for cat in by_category:
        by_category[cat].sort(key=lambda x: x["annotation_count"], reverse=True)
        
    subset = []
    # 1. Ensure at least some images from every available category
    num_categories = len(by_category)
    if num_categories == 0:
        return []
        
    images_per_cat = max(1, subset_size // num_categories)
    
    for cat, items in by_category.items():
        # Take top items by annotation count
        selected = items[:images_per_cat]
        subset.extend(selected)
        
    # 2. Fill the rest randomly from the highly-annotated remaining pool
    remaining_needed = subset_size - len(subset)
    if remaining_needed > 0:
        already_selected_ids = {item["image_id"] for item in subset}
        remaining_pool = [item for item in metadata if item["image_id"] not in already_selected_ids]
        # Sort remaining by annotation count
        remaining_pool.sort(key=lambda x: x["annotation_count"], reverse=True)
        
        # Take from the top 3x needed to allow some randomness
        top_remaining = remaining_pool[:remaining_needed * 3]
        random.seed(42)
        additional = random.sample(top_remaining, min(remaining_needed, len(top_remaining)))
        subset.extend(additional)
        
    # Remove the temporary annotation_count key
    for item in subset:
        item.pop("annotation_count", None)
        
    return subset[:subset_size]


def extract_images(zip_path: Path, extract_dir: Path, subset_metadata: list) -> None:
    """Extract only the needed images from the zip file."""
    logger.info("Extracting required images from zip...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    needed_files = {item["file_name"] for item in subset_metadata}
    extracted_count = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # The zip might have a top-level directory, so we check for ends-with
        all_members = zip_ref.namelist()
        members_to_extract = []
        
        for member in all_members:
            file_name = Path(member).name
            if file_name in needed_files:
                members_to_extract.append(member)
                
        for member in tqdm(members_to_extract, desc="Extracting"):
            file_name = Path(member).name
            target_path = extract_dir / file_name
            
            if not target_path.exists():
                with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                    target.write(source.read())
            extracted_count += 1
            
    logger.info(f"Extracted {extracted_count} images.")


def main():
    parser = argparse.ArgumentParser(description="Download and curate Fashionpedia dataset.")
    parser.add_argument("--num_images", type=int, default=TARGET_SUBSET_SIZE, help="Number of images in subset.")
    parser.add_argument("--skip_download", action="store_true", help="Skip downloading if files exist.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    zip_path = DATA_DIR / "train2020.zip"
    ann_zip_path = DATA_DIR / "instances_attributes_train2020.json"
    
    if not args.skip_download or not ann_zip_path.exists():
        download_with_progress(FASHIONPEDIA_ANNOTATIONS_URL, ann_zip_path)
        
    if not args.skip_download or not zip_path.exists():
        download_with_progress(FASHIONPEDIA_TRAIN_IMAGES_URL, zip_path)

    metadata = parse_annotations(ann_zip_path)
    logger.info(f"Total annotated images found: {len(metadata)}")
    
    subset = curate_subset(metadata, args.num_images)
    logger.info(f"Curated subset size: {len(subset)}")
    
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(subset, f, indent=2)
    logger.info(f"Saved subset metadata to {ANNOTATIONS_FILE}")
    
    extract_images(zip_path, IMAGES_DIR, subset)
    logger.info("Done!")


if __name__ == "__main__":
    main()
