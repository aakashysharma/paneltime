#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from . import callback
except:
    import callback


import sys
import os
import traceback
import datetime
import pickle
import gc
import inspect
import time
from threading import Thread
import subprocess
from queue import Queue
from queue import Empty


class Slave:
    def __init__(self, t, s_id, path, print_to_file, n_nodes):
        self.t = t
        self.id = s_id
        self.n_nodes = n_nodes
        self.d = dict()
        self.d['slave_id'] = s_id
        self.processes = Processes(self)
        while 1:
            self.wait_for_orders()

    def wait_for_orders(self):

        (msg, obj) = self.t.receive()
        response = None
        if msg == 'kill':
            sys.exit()
            response = True

        elif msg == 'callback':
            name, d = obj
            if name in self.processes:
                response = dict(self.processes[name].transact(d))
                response['slave_id'] = self.id
            else:
                response = {'empty': True}

        elif msg == 'check_state':
            response = self.processes.countalive(obj)
        elif msg == 'collect':
            response = self.processes.collect(obj)
        else:  # starting a new procedure
            self.processes[msg] = obj
        self.t.send(response)
        gc.collect()


class Processes(dict):

    def __init__(self, slave):
        self.d = slave.d
        self.parent = slave

    def __setitem__(self, name, task):
        super().__setitem__(name, Process(name, task, self))

    def countalive(self, name=None):
        if not name is None:
            if name in self:
                return self[name].thread.is_alive()
            return 0
        count = 0
        for k in self:
            if hasattr(self[k], 'thread'):
                if self[k].thread.is_alive():
                    count += 1
        return count

    def collect(self, name):
        if not name in self:
            return None, None
            # raise RuntimeError(f"The process '{name}' does not exist in node {self.parent.id}")
        d, r = self[name].collect()
        for k in d:
            self.d[k] = d[k]
        return r


class Process:
    def __init__(self, name, task, parent):
        self.name = name
        self.parent = parent
        self.task = task
        self.d = dict(parent.d)
        self.d['inbox'] = {}
        self.d['outbox'] = {}
        self.threadq = Queue()
        self.inbox = self.d['outbox']
        self.outbox = self.d['inbox']
        self.result = {}, None

        if name == 'transfer dictionary':
            t0 = time.time()
            self.get_dict(task)
        else:
            if name == 'debug':
                execute_task(self.d, task, self.threadq)
            else:
                prtask = task.replace('\n', ' ')
                # print(f"id inbox: {id(self.inbox)}, id outbox:{id(self.outbox)}, task:{prtask}")
                self.thread = Thread(target=execute_task, args=(
                    self.d, task, self.threadq, self.name))
                self.thread.start()
        a = 0

    def get_dict(self, fname):  # Blocking
        try:
            f = open(fname, 'rb')
            d = None
            d = pickle.load(f)
            f.close()
            for k in d:
                self.parent.d[k] = d[k]

        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            raise e

    def transact(self, incoming):
        # print(f'reading from {id(self.inbox)}')
        if not self.thread.is_alive():
            return dict(self.inbox)
        for k in incoming:
            self.outbox[k] = incoming[k]
        return dict(self.inbox)

    def collect(self):
        if self.threadq.empty():
            return self.result
        if not self.threadq.get() == 'start':
            raise RuntimeError("Expected a start signal")
        d, r = self.threadq.get()
        self.result = d, r
        return d, r


def execute_task(d, task, queue, name):  # None blocking
    queue.put('start')
    d = dict(d)
    try:
        r = execute_task_(d, task)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        raise e
    queue.put((d, r))


def execute_task_(d, task):
    try:  # If a returned value is desired, the task must be a function call
        return eval(task, globals(), d)
    except SyntaxError as e:  # else, it is assumed that the script only generates local variables for later use
        exec(task, globals(), d)


def run(transact, print_to_file=True):

    if not print_to_file:
        msg, (s_id, path, n_nodes), fname = handshake(transact, print_to_file)
        # Wait for instructions:
        Slave(transact, s_id, path, print_to_file, n_nodes)
        return
    try:
        msg, (s_id, path, n_nodes), fname = handshake(transact, print_to_file)
        f = open(fname, 'w', 1)
        f.close()
        # Wait for instructions:
        fstdout = os.path.join(path, f'slaves/{s_id}.txt')
        if print_to_file:
            sys.stdout = open(fstdout, 'w', 1)
        Slave(transact, s_id, path, print_to_file, n_nodes)
    except Exception as e:
        print(f'error was: {e}')
        try:
            traceback.print_exc(file=sys.stdout)
        except Exception as e:
            print(e)
        f.write('SID: %s      TIME:%s \n' % (s_id, datetime.datetime.now()))
        traceback.print_exc(file=f)
        f.flush()
        f.close()
        raise e


def handshake(transact, print_to_file):
    # Handshake:
    transact.send(os.getpid())
    path = '.'
    msg, (s_id, path, n_nodes) = transact.receive()
    transact.set('slave', n_nodes, path, s_id)
    # error handling
    fname = os.path.join(path, 'slave_errors.txt')
    return msg, (s_id, path, n_nodes), fname


def write(f, txt):
    f = open(f, 'w', 1)
    f.write(str(txt))
    f.flush()
    f.close()
