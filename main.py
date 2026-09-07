"""
Quant Finance CLI - entry shim.

Delegates to quant_finance.cli so the console script (qfcli) and
`python main.py` behave identically.
"""

import sys

from quant_finance.cli import main

if __name__ == '__main__':
    sys.exit(main())