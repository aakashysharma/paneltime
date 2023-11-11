#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This module handle callbacks


class CallBack:
    def __init__(self, inbox, outbox):
        self.inbox = inbox
        self.outbox = outbox
        self.outbox['quit'] = False

    def callback(self, **keywordargs):
        # print(f'writing to {id(self.outbox)}')
        for k in keywordargs:
            self.outbox[k] = keywordargs[k]
        return dict(self.inbox)
