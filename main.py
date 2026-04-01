#!/usr/bin/env python3
"""Python function for nth Fibonacci number using memoization"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Python function for nth Fibonacci number using memoization")
    parser.add_argument("input", help="Input file")
    parser.add_argument("-o", "--output", default="-", help="Output file")
    args = parser.parse_args()

    # TODO: implement
    print(f"Processing {args.input}")


if __name__ == "__main__":
    main()
