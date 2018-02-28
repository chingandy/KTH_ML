import monkdata as m
import dtree as dt
import random
#import drawtree_qt5 as drawt
m1=m.monk1
m2=m.monk2
m3=m.monk3

#A0
monk={1: m1,2: m2,3: m3}
mtest={1: m.monk1test, 2: m.monk2test ,3: m.monk3test}
"""
#A1
print "entropy:"
for mk in monk.keys():
	print mk,dt.entropy(monk[mk])
#A2

#A3

for mk in monk.keys():
	print "monk_"+str(mk)
	for k in range(0,6):
		entr=dt.averageGain(monk[mk], m.attributes[k])
		print entr

#A4
print "....................."

#A5

tree={}
for mk in monk.keys():
	tree[mk]=dt.buildTree(monk[mk], m.attributes)

for mk in monk.keys():
	print(dt.check(tree[mk], monk[mk]))
	print(dt.check(tree[mk], mtest[mk]))
	print (1-dt.check(tree[mk], mtest[mk]))

#A6

def partition(data, fraction):
	ldata = list(data)
	random.shuffle(ldata)
	breakPoint = int(len(ldata) * fraction)
	return ldata[:breakPoint], ldata[breakPoint:]

mdata={}
for mk in monk.keys():
	name="m"+str(mk)
	mdata[name+"train"], mdata[name+"val"] = partition(monk[mk], 0.6)

print "....................."

maxperformance=[0,0,0]
bestree=[0,0,0]
for mk in monk.keys():

	name="m"+str(mk)
	t=dt.buildTree(mdata[name+"train"], m.attributes)
	p=dt.allPruned(t)
	for pt in p:
		if(dt.check(pt, mdata[name+"val"])>maxperformance[mk-1]):
			bestree[mk-1]=pt
			maxperformance[mk-1]=dt.check(pt, mdata[name+"val"])
	print name
	print maxperformance


#drawt.drawTree(bestree[2])
"""
# A7

for i in range(0)
def partition(data, fraction):
	ldata = list(data)
	random.shuffle(ldata)
	breakPoint = int(len(ldata) * fraction)
	return ldata[:breakPoint], ldata[breakPoint:]

mdata={}
fraction=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for frc in range(0,6):
	for mk in monk.keys():
		name="m"+str(mk)
		mdata[name+"train"+str(frc)], mdata[name+"val"+str(frc)] = partition(monk[mk], fraction[frc])

print "....................."

maxperformance=[0,0,0]
bestree=[0,0,0]

mk = 1
for frc in range(0,6):
		name="m"+str(mk)
		t=dt.buildTree(mdata[name+"train"+str(frc)], m.attributes)
		p=dt.allPruned(t)
		for pt in p:
			if(dt.check(pt, mdata[name+"val"+str(frc)])>maxperformance[mk-1]):
				bestree[mk-1]=pt
				maxperformance[mk-1]=dt.check(pt, mdata[name+"val"+str(frc)])

		print name+str(frc)
		print maxperformance

mk=3
for frc in range(0,6):
		name="m"+str(mk)
		t=dt.buildTree(mdata[name+"train"+str(frc)], m.attributes)
		p=dt.allPruned(t)
		for pt in p:
			if(dt.check(pt, mdata[name+"val"+str(frc)])>maxperformance[mk-1]):
				bestree[mk-1]=pt
				maxperformance[mk-1]=dt.check(pt, mdata[name+"val"+str(frc)])

		print name+str(frc)
		print maxperformance
		print max(maxperformance)
