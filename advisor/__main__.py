"""Allow `python3 -m advisor` by delegating to advisor.main.main()."""
from .main import main

raise SystemExit(main())
