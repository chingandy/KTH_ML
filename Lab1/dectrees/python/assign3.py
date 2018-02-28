# encoding: utf-8
import math



import monkdata as m
from dtree import *
print "MONK-1:"
for i in range(0,6):
    print averageGain(m.monk1, m.attributes[i])

print "MONK-2:"
for i in range(0,6):
    print averageGain(m.monk2, m.attributes[i])

print "MONK-3:"
for i in range(0,6):
    print averageGain(m.monk3, m.attributes[i])
