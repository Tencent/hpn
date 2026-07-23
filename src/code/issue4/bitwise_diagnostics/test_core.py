import json
import struct
import tempfile
import unittest
from pathlib import Path

from core import TensorCapture, compare_capture, compare_runs


def capture(values, call=0, rank=0):
    return TensorCapture(
        call=call,
        rank=rank,
        dtype="torch.float32",
        shape=[len(values)],
        payload_hex=struct.pack(f"<{len(values)}f", *values).hex(),
    )


class CompareCaptureTests(unittest.TestCase):
    def test_identical_payload(self):
        self.assertIsNone(compare_capture(capture([1.0]), capture([1.0]), 1))

    def test_reports_first_offset_and_ulp(self):
        result = compare_capture(capture([1.0, 2.0]), capture([1.0, 2.000000238418579]), 1)
        self.assertEqual(result.first_byte, 4)
        self.assertEqual(result.first_element, 1)
        self.assertEqual(result.max_ulp_error, 1)
        self.assertGreater(result.changed_bits, 0)

    def test_negative_float_ulp_ordering(self):
        result = compare_capture(capture([-1.0]), capture([-1.0000001192092896]), 1)
        self.assertEqual(result.max_ulp_error, 1)


class CompareRunsTests(unittest.TestCase):
    def test_localizes_first_call(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for run, changed in enumerate((False, True)):
                items = [capture([1.0], call=0), capture([2.0 if not changed else 3.0], call=1)]
                data = {
                    "schema_version": 1,
                    "metadata": {
                        "op": "all_reduce", "dtype": "torch.float32", "elements": 1,
                        "calls": 2, "world_size": 1, "seed": 1,
                    },
                    "captures": [item.__dict__ for item in items],
                }
                path = Path(directory) / f"{run}.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                paths.append(path)
            report = compare_runs(paths)
            self.assertFalse(report["bitwise_identical"])
            self.assertEqual(report["first_divergence"]["call"], 1)


if __name__ == "__main__":
    unittest.main()
