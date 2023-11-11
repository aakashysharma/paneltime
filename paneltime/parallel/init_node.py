#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

try:
    from . import main
    from . import slave
except:
    import main
    import slave

slave.run(main.Transact(sys.stdin, sys.stdout), True)
