#!/usr/bin/env python
# coding: utf-8

import os
import sys

elem_list = [[16, 16]]

strings = []
for elem in elem_list:
    gene = ""
    for x, i in enumerate(elem):
        for _ in range(0, i):
            gene += str(x)
    strings.append(gene)

with open("inp.params", "w") as w:
    for i in strings:
        w.write(i + "\n")
    
