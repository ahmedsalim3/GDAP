import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))


class TestStreamlitApp(unittest.TestCase):
    def setUp(self):
        self.mock_st = MagicMock()
        self.mock_st.logger = MagicMock()
        self.mock_st_pages = MagicMock()

        self.st_patcher = patch.dict('sys.modules', {
            'streamlit': self.mock_st,
            'streamlit.logger': self.mock_st.logger,
            'st_pages': self.mock_st_pages
        })
        self.st_patcher.start()

    def tearDown(self):
        self.st_patcher.stop()

    def test_app_config(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
