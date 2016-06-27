# 8/24/2015
# Suyoun Kim (suyoun@cmu.edu)
#
import sys
import re
import fnmatch
import os
import shutil
matches = []
indir='data-fbank'
targetdir='data-fbank'

indir=sys.argv[1]
targetdir=sys.argv[2]

def multipleReplace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text

def makeDic(uttlist1, uttlist2):
	wordDict = {}
	for it in range(len(uttlist1)):
		wordDict[uttlist1[it]] = uttlist2[it]
	return wordDict

def replaceDir(directory, dic):
	#print directory
	flist = []
	for root, subFolder, files in os.walk(directory+'/'):
		for item in files:
			flist.append(root+'/'+item)
	for f in flist:
		print f
		fout = open(f+'.rep', 'w')
		for line in open(f,'r').readlines():
			newtext = multipleReplace(line, dic)
			fout.write(newtext)
		fout.close()
		shutil.move(f+'.rep', f)

def main():
	# 1. Read uttid pair of (6ch - close) 
	#    tr05 and dt05
	uttlist_input=[]
	uttlist_target=[]
	print 'Generate Dictionary'
	with open(indir+'/tr05_multi_6ch/feats.scp','r') as fe:
		for line in fe:
			uttlist_input.append(line.strip().split(' ')[0])
	with open(indir+'/dt05_real_6ch/feats.scp','r') as fe:
		for line in fe:
			uttlist_input.append(line.strip().split(' ')[0])
	with open('data/tr05_multi_close/feats.scp','r') as fe:
		for line in fe:
			uttlist_target.append(line.strip().split(' ')[0])
	with open('data/dt05_real_close/feats.scp','r') as fe:
		for line in fe:
			uttlist_target.append(line.strip().split(' ')[0])

	# 2. Make dictionary 
	dic=makeDic(uttlist_target, uttlist_input)

	# 3. Replace close uttid => 6ch uttid 
	#    tr05 and dt05
	print 'Replace uttid in target dir'
	replaceDir(targetdir+'/tr05_simu_close',dic)
	replaceDir(targetdir+'/tr05_multi_close',dic)
	replaceDir(targetdir+'/tr05_real_close',dic)
	replaceDir(targetdir+'/dt05_real_close',dic)

if __name__ == "__main__":
	main()




