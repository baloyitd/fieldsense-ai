"""
Allow ``python -m ncaa_data`` to invoke the CLI.
"""

import sys
from ncaa_data.cli import main

sys.exit(main())
