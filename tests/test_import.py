"""Development-only check that clone preserves the selected original source bytes."""
import hashlib
import json
from pathlib import Path
import unittest


class ImportTests(unittest.TestCase):
    def test_scoped_original_bytes_survive_checkout(self):
        root = Path(__file__).resolve().parents[1]
        provenance = json.loads((root / "provenance/source-import.json").read_text(encoding="utf-8"))
        self.assertEqual(len(provenance["records"]), 150)
        for row in provenance["records"]:
            path = (root / row["path"]).resolve()
            self.assertTrue(path.is_relative_to(root))
            data = path.read_bytes()
            with self.subTest(path=row["path"]):
                self.assertEqual(len(data), row["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])


if __name__ == "__main__":
    unittest.main()
