from PIL import Image
import os
import time

num = 9

c = 0
f = open(str(num)+".json", "a")   
f.write("{\n")
dirs = os.listdir("datasets/3"+str(num)+"/hsf_0/")

for a in dirs:
    f.write(",\n\n")
    s = "["
    img = Image.open('datasets/3'+str(num)+'/hsf_0/'+a).convert('L')
    for x in range(0,128,4):
        for y in range(0,128,4):
            v = 0
            for n1 in range(0,4):
                for n2 in range(0,4):
                    v += img.getpixel((x+n1,y+n2))/16;  
            
            s = s + str(int(v))

            if(not (x == 124 and y == 124)):
                s = s + ","
    s = s + "]"
    
    f.write('"' + (str(c) + '"' + ': ' + s))
    c = c + 1

    if c == 100:
        break
    

f.write("\n}\n")



