#!/usr/bin/env python3
"""
Documentation Sync Script for Berkeley Humanoid Lite

This script downloads markdown files from the Berkeley Humanoid Lite GitBook
documentation and stores them locally for offline reference and LLM grounding.

Usage:
    python sync_docs.py
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urljoin
import hashlib

try:
    import requests
except ImportError:
    print("Error: 'requests' library not found.")
    print("Please install it with: pip install requests")
    sys.exit(1)


BASE_URL = "https://berkeley-humanoid-lite.gitbook.io"
# Get the project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
LLMS_TXT = DOCS_DIR / "llms.txt"


def parse_llms_txt() -> List[str]:
    """
    Parse the llms.txt file to extract documentation paths.

    Returns:
        List of documentation paths (e.g., '/docs/home.md')

    Raises:
        FileNotFoundError: If llms.txt doesn't exist
    """
    if not LLMS_TXT.exists():
        raise FileNotFoundError(f"llms.txt not found at {LLMS_TXT}")

    paths = []
    with open(LLMS_TXT, 'r', encoding='utf-8') as f:
        for line in f:
            # Look for markdown links in the format [Title](/docs/path.md)
            matches = re.findall(r'\[.*?\]\((\/docs\/.*?\.md)\)', line)
            paths.extend(matches)

    if not paths:
        print("Warning: No documentation paths found in llms.txt")

    return paths


def get_file_hash(filepath: Path) -> str:
    """
    Calculate MD5 hash of a file.

    Args:
        filepath: Path to the file

    Returns:
        MD5 hash as hex string
    """
    if not filepath.exists():
        return ""

    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_file(url: str, output_path: Path, force: bool = False) -> Tuple[bool, str]:
    """
    Download a file from URL and save it to output_path.

    Args:
        url: URL to download from
        output_path: Local path to save the file
        force: If True, download even if file exists

    Returns:
        Tuple of (success: bool, message: str)
    """
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and get old hash
    old_hash = get_file_hash(output_path) if output_path.exists() else None

    try:
        print(f"Downloading: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save to temporary location first
        temp_path = output_path.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
            f.write(response.content)

        # Calculate new hash
        new_hash = get_file_hash(temp_path)

        # Check if content changed
        if old_hash and old_hash == new_hash and not force:
            temp_path.unlink()
            return True, "unchanged"

        # Move temp file to final location
        temp_path.replace(output_path)

        if old_hash:
            return True, "updated"
        else:
            return True, "downloaded"

    except requests.exceptions.RequestException as e:
        return False, f"failed: {str(e)}"


def sync_llms_txt(force: bool = False) -> Tuple[bool, str]:
    """
    Sync the llms.txt file from GitBook.

    Args:
        force: If True, re-download even if file exists

    Returns:
        Tuple of (success: bool, message: str)
    """
    llms_url = urljoin(BASE_URL, "llms.txt")
    print("Step 1: Syncing llms.txt index file")
    print("-" * 60)

    success, status = download_file(llms_url, LLMS_TXT, force)

    if success:
        status_symbol = "✓" if status in ['downloaded', 'updated'] else "="
        print(f"  {status_symbol} llms.txt: {status}\n")
    else:
        print(f"  ✗ llms.txt: {status}\n")

    return success, status


def sync_documentation(force: bool = False) -> None:
    """
    Sync all documentation files from GitBook.

    Args:
        force: If True, re-download all files even if they exist
    """
    print("Berkeley Humanoid Lite Documentation Sync")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"Output Directory: {DOCS_DIR}")
    print()

    # First, sync the llms.txt file
    llms_success, llms_status = sync_llms_txt(force)
    if not llms_success:
        print(f"Error: Failed to sync llms.txt: {llms_status}")
        print("Cannot proceed without the index file.")
        sys.exit(1)

    # Parse paths from llms.txt
    print("Step 2: Syncing documentation files")
    print("-" * 60)
    doc_paths = parse_llms_txt()
    print(f"Found {len(doc_paths)} documentation files to sync\n")

    # Statistics
    stats = {
        'downloaded': 0,
        'updated': 0,
        'unchanged': 0,
        'failed': 0
    }

    # Download each file
    for doc_path in doc_paths:
        # Construct URL
        # Remove leading slash and convert to GitBook URL format
        url_path = doc_path.lstrip('/')
        url = urljoin(BASE_URL, url_path)

        # Determine local output path
        # Keep the same structure as in llms.txt
        output_path = PROJECT_ROOT / doc_path.lstrip('/')

        # Download
        success, status = download_file(url, output_path, force)

        # Update statistics
        if success:
            stats[status] += 1
            status_symbol = "✓" if status in ['downloaded', 'updated'] else "="
            print(f"  {status_symbol} {doc_path.split('/')[-1]}: {status}")
        else:
            stats['failed'] += 1
            print(f"  ✗ {doc_path.split('/')[-1]}: {status}")

    # Print summary
    print("\n" + "=" * 60)
    print("Sync Summary:")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Updated:    {stats['updated']}")
    print(f"  Unchanged:  {stats['unchanged']}")
    print(f"  Failed:     {stats['failed']}")
    print(f"  Total:      {len(doc_paths)}")

    if stats['failed'] > 0:
        print("\nWarning: Some files failed to download.")
        print("Check your internet connection and try again.")
        sys.exit(1)
    else:
        print("\n✓ Documentation sync completed successfully!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync Berkeley Humanoid Lite documentation from GitBook"
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download of all files, even if they exist'
    )

    args = parser.parse_args()

    try:
        sync_documentation(force=args.force)
    except KeyboardInterrupt:
        print("\n\nSync interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
