import os


def top_folder():
  """
  Returns the absolute path to the top-level directory of this repository.

  https://stackoverflow.com/questions/3430372/how-do-i-get-the-full-path-of-the-current-files-directory
  https://stackoverflow.com/questions/9856683/using-pythons-os-path-how-do-i-go-up-one-directory
  """
  return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resources_folder(reldir=""):
  """
  Returns the absolute path of the 'resources' folder.

  reldir (str) : If specified, this directory will be joined to the resources directory.
  """
  return os.path.join(top_folder(), "resources", reldir)


def output_folder(reldir=""):
  """
  Returns the absolute path of the 'output' folder.

  reldir (str) : If specified, this directory will be joined to the resources directory.
  """
  return os.path.join(top_folder(), "output", reldir)
