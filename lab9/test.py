import unittest
from datetime import datetime, timezone
from etl import transform

SAMPLE = [
    {"userId": 1, "id": 1,  "title": "hello world", "body": "text"},
    {"userId": 2, "id": 5,  "title": "second post",  "body": "text"},
]

class TestTransform(unittest.TestCase):

    def setUp(self):
        self.result = transform(SAMPLE)

    def test_count(self):
        self.assertEqual(len(self.result), len(SAMPLE))

    def test_title_uppercase(self):
        for p in self.result:
            self.assertEqual(p["title"], p["title"].upper())

    def test_fields_present(self):
        for p in self.result:
            for field in ("userId", "id", "title", "body", "timestamp"):
                self.assertIn(field, p)

    def test_no_extra_fields(self):
        for p in self.result:
            self.assertEqual(set(p.keys()), {"userId", "id", "title", "body", "timestamp"})

    def test_timestamp_utc(self):
        for p in self.result:
            dt = datetime.fromisoformat(p["timestamp"])
            self.assertEqual(dt.tzinfo, timezone.utc)

    def test_ids_preserved(self):
        for orig, res in zip(SAMPLE, self.result):
            self.assertEqual(orig["id"], res["id"])

    def test_empty(self):
        self.assertEqual(transform([]), [])

    def test_same_timestamp(self):
        self.assertEqual(len({p["timestamp"] for p in self.result}), 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)