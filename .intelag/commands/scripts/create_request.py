"""Create timestamped agent request files from templates and copy prompt to clipboard."""

import argparse
import datetime
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = ".intelag/.agent/requests"
TIMESTAMP_FORMAT = "%y%m%d_%H_%M_%S"
DISPLAY_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
ALLOWED_FILENAME_CHARS = (" ", ".", "_", "-")
DEFAULT_FILENAME = "request"
SECTION_PATTERN_TEMPLATE = r"____________________ \[%s\]\s*\n(.*?)(?=\n____________________ \[|$)"


def create_request(
    filename: str,
    output_dir: str,
    prompt_template: str | None = None,
    template_name: str | None = None,
    template_file: str | None = None,
    extra_vars: str | None = None,
) -> str:
    """Create a timestamped request file from optional template and return its path."""
    if not filename or not filename.strip():
        logger.error("Filename is required.")
        sys.exit(1)

    filename_clean = filename.strip()

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    timestamp = now.strftime(TIMESTAMP_FORMAT)

    if filename_clean.lower().endswith(".txt"):
        filename_clean = filename_clean[:-4]

    filename_clean = "".join(c for c in filename_clean if c.isalnum() or c in ALLOWED_FILENAME_CHARS).strip()
    if not filename_clean:
        filename_clean = DEFAULT_FILENAME

    final_filename = f"{timestamp}_{filename_clean}.txt"
    full_path = output_dir_path / final_filename

    if template_name and ": " in template_name:
        parts = template_name.split(": ", 1)
        ext_file = parts[0]
        ext_name = parts[1]
        if template_file and Path(template_file).is_dir():
            template_file = str(Path(template_file) / ext_file)
            template_name = ext_name
        elif not template_file:
            template_file = ext_file
            template_name = ext_name

    template_content = ""
    if template_file and template_name:
        template_path = Path(template_file)
        if template_path.exists():
            try:
                with template_path.open(encoding="utf-8") as f:
                    content = f.read()
                pattern = SECTION_PATTERN_TEMPLATE % re.escape(template_name)
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    template_content = match.group(1).strip()
                else:
                    logger.warning(
                        "Template section '%s' not found in %s",
                        template_name,
                        template_file,
                    )
            except OSError as e:
                logger.error("Reading template file failed: %s", e)
                sys.exit(1)
            except re.error as e:
                logger.error("Regex error: %s", e)
                sys.exit(1)
        else:
            logger.warning("Template file %s not found.", template_file)

    vars_dict: dict[str, object] = {}
    env_vars_json = os.environ.get("INTELAG_VARS_JSON")
    if env_vars_json:
        try:
            vars_dict.update(json.loads(env_vars_json))
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse INTELAG_VARS_JSON: %s", e)
    if extra_vars:
        try:
            vars_dict.update(json.loads(extra_vars))
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse --vars: %s", e)

    if template_content and vars_dict:
        for key, value in vars_dict.items():
            template_content = template_content.replace(f"{{{key}}}", str(value))

    try:
        with full_path.open("w", encoding="utf-8") as f:
            if template_content:
                f.write(template_content + "\n")
            else:
                f.write(f"Request: {filename_clean}\n")
                f.write(f"Created: {now.strftime(DISPLAY_DATETIME_FORMAT)}\n")
                f.write("-" * 20 + "\n\n")
    except OSError as e:
        logger.error("Failed to create request file: %s", e)
        sys.exit(1)

    logger.info("Created request file: %s", full_path)

    if prompt_template:
        try:
            cwd = Path.cwd().resolve()
            full_path_resolved = full_path.resolve()
            try:
                rel_path = str(full_path_resolved.relative_to(cwd))
            except ValueError:
                rel_path = str(full_path_resolved)
            prompt = prompt_template.replace("{FILE_PATH}", rel_path)
            if sys.platform == "win32":
                subprocess.run(
                    "clip",
                    input=prompt.encode("utf-16"),
                    check=True,
                    shell=True,
                )
            logger.info("Agent prompt copied to clipboard.")
        except (subprocess.SubprocessError, OSError) as e:
            logger.warning("Failed to copy prompt to clipboard: %s", e)

    return str(full_path)


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info("Example: create_request('test_request', '.intelag/.agent/requests')")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Create a new request file with timestamp.")
    parser.add_argument("filename", help="Name of the request file")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument("--template", help="Template for the clipboard prompt")
    parser.add_argument(
        "--template-name",
        help="Name of the section to use from the template file",
    )
    parser.add_argument(
        "--template-file",
        help="Path to the template file containing sections",
    )
    parser.add_argument(
        "--vars",
        help="JSON string of variables to replace in the template",
    )

    args = parser.parse_args()

    try:
        create_request(
            args.filename,
            args.output,
            args.template,
            args.template_name,
            args.template_file,
            args.vars,
        )
    except Exception as e:  # pylint: disable=broad-except
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--example" in sys.argv:
        example_usage()
    else:
        main()
