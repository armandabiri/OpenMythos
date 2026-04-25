# Log Format Converter - Usage Guide

## Installation & Setup

1. Save the converter code as `log_converter.py`
2. Make it executable: `chmod +x log_converter.py`
3. Run from command line or import as a Python module

## Command Line Usage

### Basic Usage

```bash
# Preview changes without modifying files (recommended first step)
python log_converter.py /path/to/your/project --dry-run

# Actually make the changes (creates .bak backups by default)
python log_converter.py /path/to/your/project

# Process current directory
python log_converter.py . --dry-run
```

### Advanced Options

```bash
# Exclude additional directories
python log_converter.py /project --exclude tests --exclude docs --exclude examples

# Don't create backup files
python log_converter.py /project --no-backup

# Verbose output for debugging
python log_converter.py /project --verbose --dry-run

# Handle paths with spaces
python log_converter.py "/path with spaces/to project" --dry-run
```

### Real-world Examples

```bash
# Process a Django project, excluding migrations and static files
python log_converter.py /home/user/myapp --exclude migrations --exclude static --exclude media

# Process a Flask project with custom exclusions
python log_converter.py /opt/flask_app --exclude instance --exclude uploads --dry-run

# Process with no backups (use with caution!)
python log_converter.py /tmp/test_project --no-backup --verbose
```

## What Gets Converted

### F-string Logging (Before → After)

```python
# Before: F-string logging (eager evaluation)
logger.info(f"Processing file {filename} with {count} records")
logger.error(f"Failed to connect to {host}:{port}")
self.logger.debug(f"User {user_id} performed action {action}")

# After: Lazy % formatting (deferred evaluation)
logger.info("Processing file %s with %s records", filename, count)
logger.error("Failed to connect to %s:%s", host, port)
self.logger.debug("User %s performed action %s", user_id, action)
```

### .format() Logging (Before → After)

```python
# Before: .format() method (eager evaluation)
logger.warning("Status: {} for item {}".format(status, item_id))
log.info("Connected to {} successfully".format(database_url))

# After: Lazy % formatting (deferred evaluation)
logger.warning("Status: %s for item %s", status, item_id)
log.info("Connected to %s successfully", database_url)
```

### Named Placeholders

```python
# Before: Named format placeholders
logger.info("User {username} logged in from {ip}".format(username=user, ip=client_ip))

# After: Named % formatting
logger.info("User %(username)s logged in from %(ip)s", {"username": user, "ip": client_ip})
```

## Sample Output

### Dry Run Output
```
DRY RUN: Processing directory: /home/user/myproject
Excluding directories: .git, .venv, __pycache__, build, dist, node_modules, tests

Found 23 Python files to process

/home/user/myproject/app/models.py:45
  OLD: logger.info(f"Creating user {username} with email {email}")
  NEW: logger.info("Creating user %s with email %s", username, email)

/home/user/myproject/app/views.py:112
  OLD: log.error(f"Database connection failed: {str(e)}")
  NEW: log.error("Database connection failed: %s", str(e))

Summary:
  Files processed: 5
  Total changes: 12
  (This was a dry run - no files were actually modified)
```

### Actual Conversion Output
```
Processing directory: /home/user/myproject
Excluding directories: .git, .venv, __pycache__, build, dist

Found 23 Python files to process

Backup created: /home/user/myproject/app/models.py.bak
Modified /home/user/myproject/app/models.py: 3 changes
Backup created: /home/user/myproject/app/views.py.bak
Modified /home/user/myproject/app/views.py: 7 changes

Summary:
  Files processed: 5
  Total changes: 12
```

## Python Module Usage

```python
from log_converter import LogFormatConverter

# Create converter with custom settings
converter = LogFormatConverter(
    exclude_dirs={'custom_dir', 'temp_files'},
    backup=True,        # Create .bak files
    dry_run=False      # Actually make changes
)

# Process a directory
converter.convert_directory('/path/to/project')

# Check results
print(f"Files modified: {converter.files_processed}")
print(f"Total changes: {converter.changes_made}")
```

## Using the log_lazy Helper Function

```python
from log_converter import log_lazy
import logging

logger = logging.getLogger(__name__)

# Use log_lazy for immediate lazy formatting
log_lazy(logger, logging.INFO, f"Processing {filename}")
log_lazy(logger, logging.ERROR, "Failed with status {}", status_code)
log_lazy(logger, logging.DEBUG, "User {user} action {action}", user=username, action=user_action)
```

## Directory Structure Example

```
my_project/
├── app/
│   ├── models.py          # ✓ Will be processed
│   ├── views.py           # ✓ Will be processed
│   └── __pycache__/       # ✗ Excluded automatically
├── tests/                 # ✗ Can exclude with --exclude tests
├── .venv/                 # ✗ Excluded automatically
├── .git/                  # ✗ Excluded automatically
├── requirements.txt       # ✗ Not a Python file
└── manage.py              # ✓ Will be processed
```

## Safety Features

1. **Automatic Backups**: Original files saved as `.bak` before modification
2. **Dry Run Mode**: Preview all changes before applying them
3. **Safe Exclusions**: Automatically skips sensitive directories
4. **Error Handling**: Graceful handling of encoding issues and file access problems
5. **Progress Reporting**: Clear feedback on what's being processed

## Best Practices

1. **Always start with --dry-run** to preview changes
2. **Test on a small directory first** before processing large codebases
3. **Keep backups enable** unless you're absolutely sure
4. **Use version control** - commit your code before running the converter
5. **Review changes** after conversion to ensure correctness

## Troubleshooting

### Common Issues

```bash
# Permission denied
sudo python log_converter.py /protected/directory --dry-run

# Path with spaces
python log_converter.py "/path with spaces" --dry-run

# Large project - exclude more directories
python log_converter.py /huge/project --exclude node_modules --exclude .tox --exclude htmlcov
```

### Verification

After conversion, you can verify the changes work correctly:

```python
# Test that your logging still works
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# These should all work after conversion
logger.info("Simple message")
logger.info("Message with %s", "parameter")
logger.info("Message with %(name)s", {"name": "value"})
```