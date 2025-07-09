import os
import re
from typing import List, Dict

# Configuration
DOCS_DIR: str = "docs"
THUMBNAIL_DIR: str = os.path.join(DOCS_DIR, "plots", "thumbnails")
GIF_DIR: str = os.path.join(DOCS_DIR, "plots", "gallery")
GALLERY_EXAMPLES_DIR: str = os.path.join(DOCS_DIR, "gallery_examples")
GALLERY_MD_PATH: str = os.path.join(DOCS_DIR, "gallery.md")

# Acronyms to keep in uppercase
ACRONYMS: List[str] = ["SGD", "EMA", "DBSCAN", "RANSAC", "KMeans"]

def prettify_name(filename: str) -> str:
    """
    Converts a filename like 'sklearn_regression_lasso' into 'Regression Lasso'.

    Args:
        filename: The input filename to process.

    Returns:
        A prettified string with proper capitalization and spacing.
    """
    # Remove extension and thumbnail suffix
    base_name = filename.replace("_thumbnail.png", "")

    # Remove library prefix for cleaner titles
    prefixes = ["sega_learn_", "sklearn_"]
    for prefix in prefixes:
        if base_name.startswith(prefix):
            base_name = base_name[len(prefix):]
            break

    # Split by underscore and handle camelCase
    parts = base_name.replace("_", " ").split()
    expanded_parts = []
    for part in parts:
        sub_parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', part).split()
        expanded_parts.extend(sub_parts)

    # Capitalize words and preserve specific acronyms
    pretty_parts = [
        part.upper() if part.upper() in ACRONYMS else part.capitalize()
        for part in expanded_parts
    ]

    return " ".join(pretty_parts)

def generate_individual_pages() -> None:
    """
    Generates a markdown file for each animation in the gallery_examples directory.
    """
    os.makedirs(GALLERY_EXAMPLES_DIR, exist_ok=True)
    print("Generating individual gallery pages...")

    for thumbnail_file in sorted(os.listdir(THUMBNAIL_DIR)):
        if not thumbnail_file.endswith("_thumbnail.png"):
            continue

        base_name = thumbnail_file.replace("_thumbnail.png", "")
        gif_file = f"{base_name}.gif"
        md_file = f"{base_name}.md"

        if not os.path.exists(os.path.join(GIF_DIR, gif_file)):
            print(f"  [!] Warning: GIF not found for {thumbnail_file}. Skipping.")
            continue

        pretty_title = prettify_name(base_name)
        md_content = f"""# {pretty_title}

<!-- This page is automatically generated. Do not edit manually. -->

![{pretty_title} Animation](../../plots/gallery/{gif_file})

Back to the [Gallery Index](../gallery.md).
"""
        with open(os.path.join(GALLERY_EXAMPLES_DIR, md_file), "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"  -> Generated: {md_file}")

def generate_gallery_index() -> None:
    """
    Generates the main gallery.md file with sections for each library.
    """
    print("\nGenerating gallery index page...")

    # Group files by library prefix
    grouped_files: Dict[str, List[str]] = {
        "sklearn": [],
        "sega_learn": [],
        "other": []
    }

    for filename in sorted(os.listdir(THUMBNAIL_DIR)):
        if not filename.endswith("_thumbnail.png"):
            continue
        if filename.startswith("sklearn_"):
            grouped_files["sklearn"].append(filename)
        elif filename.startswith("sega_learn_"):
            grouped_files["sega_learn"].append(filename)
        else:
            grouped_files["other"].append(filename)

    # Build markdown content
    gallery_md_content = """# Animation Gallery

<!-- This page is automatically generated. Do not edit manually. -->

Welcome to the `ez-animate` gallery! Our animations are compatible with both Scikit-learn and `sega_learn` models. Browse the examples below to see what you can create.

---
"""
    section_map = {
        "sklearn": "Scikit-learn Examples",
        "sega_learn": "Sega-learn Examples"
    }

    for library, title in section_map.items():
        thumbnail_files = grouped_files.get(library, [])
        if not thumbnail_files:
            continue

        gallery_md_content += f"\n## {title}\n\n<ul class=\"grid cards columns-2\" markdown>\n"

        for thumbnail_file in thumbnail_files:
            base_name = thumbnail_file.replace("_thumbnail.png", "")
            pretty_title = prettify_name(base_name)
            card_html = f"""  <li>
    <a href="../gallery_examples/{base_name}/" class="card">
      <div class="card__image">
        <img src="../plots/thumbnails/{thumbnail_file}" alt="{pretty_title} Thumbnail">
      </div>
      <div class="card__content">
        <p><strong>{pretty_title}</strong></p>
      </div>
    </a>
  </li>
"""
            gallery_md_content += card_html

        gallery_md_content += "</ul>\n"

    with open(GALLERY_MD_PATH, "w", encoding="utf-8") as f:
        f.write(gallery_md_content)
    print(f"  -> Generated: {os.path.basename(GALLERY_MD_PATH)}")

def main() -> None:
    """Main function to generate gallery pages and index."""
    generate_individual_pages()
    generate_gallery_index()
    print("\nGallery generation complete.")
    print(f"Generated files in '{GALLERY_EXAMPLES_DIR}' and '{GALLERY_MD_PATH}'.")

if __name__ == "__main__":
    main()
