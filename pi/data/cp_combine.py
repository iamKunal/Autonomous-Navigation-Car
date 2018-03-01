#!/usr/bin/python2

lst = []

with open('lst.txt', 'r') as f:
	lst = f.read().split('\n')

i=1

for img in lst:
	if 'resized' in img:
		with open(img, 'r') as rd:
			with open("combined/%05d.%s.png" % (i, img.split('.')[1]), 'w') as wr:
				wr.write(rd.read())
		i+=1

