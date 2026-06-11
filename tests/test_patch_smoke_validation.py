import unittest

from cicl_agent.evaluation.patch_smoke_validation import (
    _call_qdp_line_type,
    _exercise_non_inplace_replace,
    _exercise_rst,
)


class PatchSmokeValidationTest(unittest.TestCase):
    def test_qdp_line_type_detects_lowercase_command_fix(self):
        base = '''
def _line_type(line, delimiter=None):
    import re
    _line_type_re = re.compile(r"^READ SERR(\\s+[0-9]+)+$")
    if _line_type_re.match(line.strip()) is None:
        raise ValueError(f"Unrecognized QDP line: {line.strip()}")
    return "command"
'''
        patched = '''
def _line_type(line, delimiter=None):
    import re
    _line_type_re = re.compile(r"^READ SERR(\\s+[0-9]+)+$", re.IGNORECASE)
    if _line_type_re.match(line.strip()) is None:
        raise ValueError(f"Unrecognized QDP line: {line.strip()}")
    return "command"
'''

        self.assertFalse(_call_qdp_line_type(base, "read serr 1 2")["ok"])
        self.assertEqual("command", _call_qdp_line_type(patched, "read serr 1 2")["value"])

    def test_rst_slice_exercises_header_rows_read_and_write(self):
        base = '''
class RST(FixedWidth):
    data_class = SimpleRSTData
    header_class = SimpleRSTHeader

    def __init__(self):
        super().__init__(delimiter_pad=None, bookend=False)

    def write(self, lines):
        lines = super().write(lines)
        lines = [lines[1]] + lines + [lines[1]]
        return lines
'''
        patched = '''
class RST(FixedWidth):
    data_class = SimpleRSTData
    header_class = SimpleRSTHeader

    def __init__(self, header_rows=None):
        super().__init__(delimiter_pad=None, bookend=False, header_rows=header_rows)

    def read(self, table):
        n_header = len(self.header.header_rows) if self.header.header_rows else 1
        self.data.start_line = 2 + n_header
        return super().read(table)

    def write(self, lines):
        lines = super().write(lines)
        idx = len(self.header.header_rows) if self.header.header_rows else 1
        lines = [lines[idx]] + lines + [lines[idx]]
        return lines
'''
        self.assertFalse(_exercise_rst(base)["init_with_header_rows"]["ok"])
        result = _exercise_rst(patched)
        self.assertTrue(result["init_with_header_rows"]["ok"])
        self.assertEqual(4, result["read_start_line_after_read"])
        self.assertEqual("unit-border", result["write_border_index_value"])

    def test_non_inplace_replace_requires_assignment(self):
        result = _exercise_non_inplace_replace()
        self.assertEqual("1.0E+00", result["pre_after_replace"])
        self.assertEqual("1.0D+00", result["post_after_assignment"])


if __name__ == "__main__":
    unittest.main()
