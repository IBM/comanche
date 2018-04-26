#!/usr/bin/python
import os

comancheRoot = os.getcwd()

def gen(rootDir):
    for dirName, subdirList, fileList in os.walk(rootDir):        
        for fname in os.listdir('.'):
            if fname == 'CMakeLists.txt':
                try:
                    file = open(dirName + "/.cmakeinclude","w+")
                    file.truncate(0)
                    file.write("set(ENV{COMANCHE_HOME} " + comancheRoot + ")\n")
                    file.write("include($ENV{COMANCHE_HOME}/mk/common.cmake)\n")
                    break
                except:
                    print('Cannot access %s' % dirName)

gen('./src')
gen('./apps')
gen('./comanche-restricted/src')



