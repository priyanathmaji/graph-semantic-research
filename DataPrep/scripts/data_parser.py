# data_parser.py

import pandas as pd
from typing import Iterator, Dict, Any

def parse_amazon_data_improved(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    Parses the Amazon metadata file and yields a dictionary for each product.
    Args:
        file_path: The path to the Amazon metadata file.
    Yields:
        A dictionary containing the parsed data for a single product.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        current_product = {}
        for line in f:
            line = line.strip()

            if line.startswith('Id:'):
                if current_product:
                    yield current_product
                current_product = {'Id': line.split(':')[-1].strip(), 'is_discontinued': False}
            
            elif line == 'discontinued product':
                current_product['is_discontinued'] = True
            
            elif line.startswith('ASIN:'):
                current_product['ASIN'] = line.split(':')[-1].strip()
            elif line.startswith('title:'):
                current_product['title'] = line.split(':', 1)[-1].strip()
            elif line.startswith('group:'):
                current_product['group'] = line.split(':')[-1].strip()
            elif line.startswith('salesrank:'):
                try:
                    current_product['salesrank'] = int(line.split(':')[-1].strip())
                except ValueError:
                    current_product['salesrank'] = -1
            
            # **CORRECTED LOGIC FOR 'similar' FIELD**
            elif line.startswith('similar:'):
                parts = line.split()
                # Slice from index 2 to skip 'similar:' and the item count
                if len(parts) > 2:
                    current_product['similar'] = parts[2:]
                else:
                    current_product['similar'] = []

            elif line.startswith('categories:'):
                # This field requires more complex parsing not covered yet.
                # For now, we'll just store the count.
                try:
                    current_product['categories'] = int(line.split(':')[-1].strip())
                except ValueError:
                    current_product['categories'] = 0
            elif line.startswith('reviews:'):
                parts = line.split()
                try:
                    current_product['avg_rating'] = float(parts[-1])
                    current_product['reviews_count'] = int(parts[2])
                except (ValueError, IndexError):
                    current_product['avg_rating'] = 0.0
                    current_product['reviews_count'] = 0

        if current_product:
            yield current_product

if __name__ == "__main__":
    raw_data_path = 'data/amazon-meta.txt'
    output_parquet_path = 'data/amazon_products.parquet'

    print(f"Starting parsing of {raw_data_path}...")
    
    product_generator = parse_amazon_data_improved(raw_data_path)
    df = pd.DataFrame(product_generator)

    print(f"Parsing complete. Found {len(df)} products.")
    print(f"Found {df['is_discontinued'].sum()} discontinued products.")
    
    print(f"Saving DataFrame to {output_parquet_path}...")
    df.to_parquet(output_parquet_path, index=False)
    
    print("Done.")