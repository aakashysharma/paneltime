#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import os
import re
import subprocess as sp
import sys
import glob


def main():

    push = '-p' in sys.argv
    try:
        nukedir('dist')
        nukedir('build')
        nukedir('paneltime.egg-info')
    except FileNotFoundError:
        pass
    if push:
        version = add_version()
        gitpush(version)

    wd = os.getcwd()

    pythonfile = 'python'
    if '37' in sys.argv:
        pythonfile = '"C:\\Program Files\\Python37\\python.exe"'
    if '38' in sys.argv:
        pythonfile = '"C:\\Program Files\\Python38\\python.exe"'
    if '311' in sys.argv:
        pythonfile = '"C:\\Program Files\\Python311\\python.exe"'

    os.system(
        f'{pythonfile} {os.path.join(wd, "setup.py")} bdist_wheel sdist build')

    copy_pyd_files()

    if push:
        os.system("twine upload dist/*")


def copy_pyd_files():
    # Search for .pyd files in the build directory
    pyd_files = glob.glob(os.path.join('build', '**', '*.pyd'), recursive=True)

    # copy to assoscated dir
    for f in pyd_files:
        name = os.path.basename(f).split('.')[0] + '.pyx'
        for root, dirs, files in os.walk('paneltime'):
            if name in files:
                # Copy .pyd file to the .pyx file's directory
                shutil.copy(f, root)
                print(f"Copied {f} to {root}")


def gitpush(version):
    print(f"Packaging paneltime version {version}")
    r = sp.check_output('git pull')
    if r != b'Already up to date.\n':
        raise RuntimeError(
            f'Not up to date after git pull. Fix any conflicts and check that the repository is up to date\nPull output:\n{r})')
    os.system('git add .')
    os.system(
        f'git commit -m "New version {version} committed: {input("Write reason for commit: ")}"')
    os.system('git push')


def add_version():
    f = open('setup.py', 'r')
    s = f.read()
    m = re.search("(?<=version=')(.*)(?=')", s)
    v = s[m.start(0):m.end(0)]
    v = v.split('.')
    v = v[0], v[1], str(int(v[2])+1)
    version = '.'.join(v)
    s = s[:m.start(0)] + version + s[m.end(0):]
    save('setup~.py', s)
    save('setup.py', s)
    os.remove('setup~.py')
    save('paneltime/info.py', f"version='{version}'")
    return version


def save(file, string):
    f = open(file, 'w')
    f.write(string)
    f.close()


def rm(fldr):
    try:
        shutil.rmtree(fldr)
    except Exception as e:
        print(e)


def nukedir(dir):
    if dir[-1] == os.sep:
        dir = dir[:-1]
    if os.path.isfile(dir):
        return
    files = os.listdir(dir)
    for file in files:
        if file == '.' or file == '..':
            continue
        path = dir + os.sep + file
        if os.path.isdir(path):
            nukedir(path)
        else:
            os.unlink(path)
    os.rmdir(dir)


main()
