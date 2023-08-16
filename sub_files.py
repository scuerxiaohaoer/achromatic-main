# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:22:32 2020

@author: Administrator
"""
import os
import sys


class Tee():

  def __init__(self, name):
    self.file = open(name, 'w')
    self.stdout = sys.stdout
    self.stderr = sys.stderr
    sys.stdout = self
    sys.stderr = self

  def __del__(self):
    self.file.close()

  def write(self, data):
    self.file.write(data)
    self.stdout.write(data)
    self.file.flush()
    self.stdout.flush()

  def write_to_file(self, data):
    self.file.write(data)

  def flush(self):
    pass