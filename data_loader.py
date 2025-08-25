import pdfplumber
import pandas as pd
from collections import defaultdict, Counter
import os
from config import PDF_PATH, OUTPUT_LOG, OUTPUT_JSON_CLEAN

class DataLoader:
    def __init__(self, pdf_path=PDF_PATH, output_log=OUTPUT_LOG, output_json=OUTPUT_JSON_CLEAN):
        """
        Initialize DataLoader with file paths.
        
        Args:
            pdf_path: Path to PDF file.
            output_log: Path to log file.
            output_json: Path to output JSON file.
        """
        self.pdf_path = pdf_path
        self.output_log = output_log
        self.output_json = output_json

    def load_and_process_pdf(self):
        """
        Load and process PDF, extract text, and save to JSON.
        
        Returns:
            pandas DataFrame with page number and context.
        """
        data = []
        with pdfplumber.open(self.pdf_path) as pdf, open(self.output_log, "w", encoding="utf-8") as f:
            for page_num, page in enumerate(pdf.pages, start=1):
                f.write(f"\n===== PAGE {page_num} =====\n")
                chars = page.chars

                lines = defaultdict(list)
                for ch in chars:
                    lines[round(ch["top"])].append(ch)

                page_text = []
                for top in sorted(lines.keys()):
                    f.write(f"\n--- Line at top={top} ---\n")
                    line_chars = sorted(lines[top], key=lambda c: c["x0"])

                    deltas, ratios = [], []
                    for i in range(1, len(line_chars)):
                        delta_x = line_chars[i]["x0"] - line_chars[i-1]["x1"]
                        deltas.append(round(delta_x, 2))
                        ratio = delta_x / line_chars[i]["size"] if line_chars[i]["size"] else 0
                        ratios.append(round(ratio, 2))

                    mode_delta = Counter(deltas).most_common(1)[0][0] if deltas else 0
                    mode_ratio = Counter(ratios).most_common(1)[0][0] if ratios else 0
                    f.write(f"mode_delta={mode_delta:.2f}, mode_ratio={mode_ratio:.2f}\n")

                    last = None
                    line_text = []
                    for i, ch in enumerate(line_chars):
                        if last:
                            delta_x = ch["x0"] - last["x1"]
                            ratio = delta_x / ch["size"] if ch["size"] else 0
                            font = ch.get("fontname", "").lower()
                            diff_delta = abs(delta_x - mode_delta)
                            diff_ratio = abs(ratio - mode_ratio)
                            f.write(
                                f"'{last['text']}' -> '{ch['text']}' | font={font} | size={ch['size']:.1f} "
                                f"| delta_x={delta_x:.2f} (diff={diff_delta:.2f}) "
                                f"| ratio={ratio:.2f} (diff={diff_ratio:.2f})\n"
                            )
                            if diff_delta > 0.1 or diff_ratio > 0.1:
                                line_text.append(" ")
                        line_text.append(ch["text"])
                        last = ch
                    page_text.append("".join(line_text))

                text = "\n".join(page_text).strip()
                if text:
                    data.append({"page": page_num, "context": text})

        print(f"✅ Log saved to: {self.output_log}")
        df = pd.DataFrame(data)
        print(df.head())
        df.to_json(self.output_json, orient="records", force_ascii=False, indent=4)
        print(f"✅ Data saved to: {self.output_json}")
        return df