from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_report_tables.py"
SPEC = importlib.util.spec_from_file_location("export_report_tables", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EXPORT_REPORT_TABLES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT_REPORT_TABLES)


class ExportReportTablesTests(unittest.TestCase):
    def test_latex_main_table_highlights_current_maxima(self) -> None:
        rows = [
            {
                "system": "fixed_128",
                "exact_match": 0.90,
                "f1": 0.95,
                "recall_at_k": 0.80,
                "precision_at_k": 0.70,
                "avg_chunk_tokens": 120.0,
                "num_chunks": 10,
            },
            {
                "system": "semantic_256",
                "exact_match": 0.80,
                "f1": 0.85,
                "recall_at_k": 0.90,
                "precision_at_k": 0.72,
                "avg_chunk_tokens": 140.0,
                "num_chunks": 8,
            },
        ]

        table = EXPORT_REPORT_TABLES.latex_main_table("squad_v2", rows, "Caption", "tab:test")

        self.assertIn(r"\texttt{fixed\_128} & \textbf{90.0} & \textbf{95.0}", table)
        self.assertIn(r"\textbf{90.0}", table)
        self.assertIn(r"\textbf{72.0}", table)


if __name__ == "__main__":
    unittest.main()
