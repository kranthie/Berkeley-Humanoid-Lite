# Berkeley Humanoid Lite Documentation Archive

This directory contains a local copy of the Berkeley Humanoid Lite documentation, synced from the official GitBook documentation at https://berkeley-humanoid-lite.gitbook.io.

## Purpose

This local documentation archive serves two purposes:

1. **Offline Reference**: Access documentation without an internet connection
2. **LLM Grounding**: Provide context for AI coding assistants working on this project

## Usage

### Syncing Documentation

To download or update the documentation:

```bash
# Navigate to the docs directory
cd .external-libs/berkeley-humanoid-lite-docs

# Basic sync (downloads new files, updates changed files)
uv run sync-docs

# Or use it from anywhere in the project
uv run --directory .external-libs/berkeley-humanoid-lite-docs sync-docs

# Force re-download all files
uv run sync-docs --force
```

### Requirements

This project uses `uv` for dependency management. The `requests` library is automatically installed when you run the script with `uv run`.

## Directory Structure

```
.external-libs/berkeley-humanoid-lite-docs/
├── pyproject.toml        # Project configuration and dependencies
├── README.md             # This file
├── src/
│   └── berkeley_humanoid_lite_docs/
│       └── sync_docs.py  # Documentation sync script
└── docs/                 # Documentation files (gitignored)
    ├── llms.txt          # Index of all documentation pages
    ├── home.md
    ├── releases.md
    ├── getting-started-with-hardware/
    ├── getting-started-with-software/
    └── in-depth-contents/
```

## How It Works

1. **Syncs Index File**: First downloads/updates `docs/llms.txt` from GitBook
2. **Parses Index**: Reads `llms.txt` to find all documentation pages
3. **Downloads Files**: For each page, downloads the markdown file from GitBook
4. **Smart Sync**: Uses MD5 hashing to detect changes
   - Downloads new files
   - Updates modified files
   - Skips unchanged files (saves bandwidth and time)
5. **Maintains Structure**: Files are saved with the same directory structure as in the index

## Updating the Documentation

The documentation is maintained on GitBook. To sync the latest version:

```bash
cd .external-libs/berkeley-humanoid-lite-docs
uv run sync-docs
```

The script will:
1. First sync the `llms.txt` index file (shows new docs added to GitBook)
2. Then sync all documentation markdown files
3. Show a summary of:
   - New files downloaded
   - Existing files updated
   - Unchanged files
   - Any failures

### Git Tracking

The sync script and configuration are tracked in git, but the downloaded documentation files in `docs/` are gitignored. This means:
- The script itself is version controlled
- Downloaded docs are local only (regenerated on each machine)
- Run `uv run sync-docs` after cloning to get the latest docs

## Notes

- This is a local copy for development convenience
- The canonical source is always the GitBook documentation
- Run the sync script periodically to stay up to date
- The `.external-libs` directory is typically not committed to git
