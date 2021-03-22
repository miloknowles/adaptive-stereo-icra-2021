# Copyright 2020 Massachusetts Institute of Technology
#
# @file path_utils.py
# @author Milo Knowles
# @date 2020-09-22 23:01:51 (Tue)

import os


def path_to_toplevel():
  """
  Returns the absolute path to the top-level directory of this repository.

  https://stackoverflow.com/questions/3430372/how-do-i-get-the-full-path-of-the-current-files-directory
  https://stackoverflow.com/questions/9856683/using-pythons-os-path-how-do-i-go-up-one-directory
  """
  return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path_to_resources(reldir=""):
  """
  Returns the absolute path of the 'resources' folder.

  reldir (str) : If specified, this directory will be joined to the resources directory.
  """
  return os.path.join(path_to_toplevel(), "resources", reldir)


def path_to_output(reldir=""):
  """
  Returns the absolute path of the 'output' folder.

  reldir (str) : If specified, this directory will be joined to the resources directory.
  """
  return os.path.join(path_to_toplevel(), "output", reldir)
