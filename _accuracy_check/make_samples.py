"""
Write the sample resumes out as real .docx files and the job description as a
.txt, into ../sample_data/. These are the files to upload into the app itself.

    python _accuracy_check/make_samples.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docx import Document                                   # noqa: E402
from sample_texts import RESUMES, JOB_DESCRIPTION, JD_FILENAME  # noqa: E402

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data"
)


def main():
    os.makedirs(OUT, exist_ok=True)

    for filename, text in RESUMES.items():
        doc = Document()
        for line in text.splitlines():
            # A blank paragraph keeps the section breaks the parser relies on
            # when it decides whether a date range sits on an education line.
            doc.add_paragraph(line)
        doc.save(os.path.join(OUT, filename))
        print(f"  wrote {filename}")

    jd_path = os.path.join(OUT, JD_FILENAME)
    with open(jd_path, "w", encoding="utf-8") as fh:
        fh.write(JOB_DESCRIPTION)
    print(f"  wrote {JD_FILENAME}")
    print(f"\nAll files are in: {OUT}")


if __name__ == "__main__":
    main()
