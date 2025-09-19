#!/usr/bin/env python
import argparse, pickle, os

def merge_pickles(input_files, output_file):
    merged = {}
    for file in input_files:
        if os.path.exists(file):
            with open(file, "rb") as f:
                data = pickle.load(f)
            # If keys overlap, later files override earlier ones.
            merged.update(data)
        else:
            print(f"File {file} not found, skipping.")
    with open(output_file, "wb") as f:
        pickle.dump(merged, f)
    print(f"Merged pickle saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge multiple pickle files into one")
    parser.add_argument("--input", nargs="+", required=True,
                        help="List of pickle files to merge")
    parser.add_argument("--output", required=True,
                        help="Name of the merged pickle file")
    args = parser.parse_args()
    
    merge_pickles(args.input, args.output)

if __name__ == "__main__":
    main()
