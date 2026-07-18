import json
import random
from pathlib import Path
import argparse

def write_split(data, output_path, images_dir):
    prompt_text = (
        "Describe this fashion image in detail. Include:\n"
        "1) All visible clothing items with their exact colors, patterns, and fabric type\n"
        "2) The setting or environment (indoor/outdoor, office, street, park, home, event, etc.)\n"
        "3) The overall style or vibe (formal, casual, sporty, elegant, streetwear, etc.)\n"
        "4) Any accessories (bags, jewelry, hats, shoes, etc.)\n"
        "Be specific about colors (e.g., 'navy blue' not just 'blue') and patterns "
        "(e.g., 'pinstripe' not just 'striped')."
    )

    count = 0
    with open(output_path, 'w') as f:
        for item in data:
            target_text = item.get("structured_caption", "")
            if not target_text:
                continue

            remote_image_path = f"/data/{images_dir}/{item['file_name']}"

            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": remote_image_path},
                            {"type": "text", "text": prompt_text}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": target_text}
                        ]
                    }
                ],
                "ground_truth_categories": item.get("categories", []),
                "ground_truth_attributes": item.get("attributes", [])
            }
            f.write(json.dumps(record) + "\n")
            count += 1
    print(f"Prepared {count} examples and saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for Qwen3-VL fine-tuning")
    parser.add_argument("--annotations", type=str, default="data/annotations.json")
    parser.add_argument("--images_dir", type=str, default="images") # Relative to remote /data
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    
    if not annotations_path.exists():
        print(f"Error: {annotations_path} does not exist.")
        return

    with open(annotations_path, 'r') as f:
        metadata = json.load(f)
        
    print(f"Total metadata size: {len(metadata)}")
    
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(metadata)
    
    # Partitions: SFT (5000), GSPO (1500), Val (1000), Test (1000)
    sft_data = metadata[:5000]
    gspo_data = metadata[5000:6500]
    val_data = metadata[6500:7500]
    test_data = metadata[7500:8500]
    
    write_split(sft_data, "data/sft_data.jsonl", args.images_dir)
    write_split(gspo_data, "data/gspo_data.jsonl", args.images_dir)
    write_split(val_data, "data/val_data.jsonl", args.images_dir)
    write_split(test_data, "data/test_data.jsonl", args.images_dir)

if __name__ == "__main__":
    main()
