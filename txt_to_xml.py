#!/usr/bin/env python  
# -*- coding: UTF-8 -*-  
# txt_to_xml.py  
# According to a given XML Schema, generate an XML from a blank file in the form of a DOM tree  
from xml.dom.minidom import Document  
import cv2  
import os  

def generate_xml(name,split_lines,img_size,class_ind):  
    doc = Document()  # Create DOM document object  

    annotation = doc.createElement('annotation')  
    doc.appendChild(annotation)  

    title = doc.createElement('folder')  
    title_text = doc.createTextNode('VOC2007')#Here modified the folder name  
    title.appendChild(title_text)  
    annotation.appendChild(title)  

    img_name=name+'.jpg'#To use jpg format  

    title = doc.createElement('filename')  
    title_text = doc.createTextNode(img_name)  
    title.appendChild(title_text)  
    annotation.appendChild(title)  

    source = doc.createElement('source')  
    annotation.appendChild(source)  

    title = doc.createElement('database')  
    title_text = doc.createTextNode('The VOC2007 Database')#Modify to VOC  
    title.appendChild(title_text)  
    source.appendChild(title)  

    title = doc.createElement('annotation')  
    title_text = doc.createTextNode('PASCAL VOC2007')#Modify to VOC  
    title.appendChild(title_text)  
    source.appendChild(title)  

    size = doc.createElement('size')  
    annotation.appendChild(size)  

    title = doc.createElement('width')  
    title_text = doc.createTextNode(str(img_size[1]))  
    title.appendChild(title_text)  
    size.appendChild(title)  

    title = doc.createElement('height')  
    title_text = doc.createTextNode(str(img_size[0]))  
    title.appendChild(title_text)  
    size.appendChild(title)  

    title = doc.createElement('depth')  
    title_text = doc.createTextNode(str(img_size[2]))  
    title.appendChild(title_text)  
    size.appendChild(title)  

    for split_line in split_lines:  
        line=split_line.strip().split()  
        if line[0] in class_ind:  
            object = doc.createElement('object')  
            annotation.appendChild(object)  

            title = doc.createElement('name')  
            title_text = doc.createTextNode(line[0])  
            title.appendChild(title_text)  
            object.appendChild(title)  

            title = doc.createElement('difficult')  
            title_text = doc.createTextNode('0')  
            title.appendChild(title_text)  
            object.appendChild(title)  

            bndbox = doc.createElement('bndbox')  
            object.appendChild(bndbox)  
            title = doc.createElement('xmin')  
            title_text = doc.createTextNode(str(int(float(line[4]))))  
            title.appendChild(title_text)  
            bndbox.appendChild(title)  
            title = doc.createElement('ymin')  
            title_text = doc.createTextNode(str(int(float(line[5]))))  
            title.appendChild(title_text)  
            bndbox.appendChild(title)  
            title = doc.createElement('xmax')  
            title_text = doc.createTextNode(str(int(float(line[6]))))  
            title.appendChild(title_text)  
            bndbox.appendChild(title)  
            title = doc.createElement('ymax')  
            title_text = doc.createTextNode(str(int(float(line[7]))))  
            title.appendChild(title_text)  
            bndbox.appendChild(title)  

    # Write DOM object doc to file  
    f = open('Annotations/'+name+'.xml','w')  
    f.write(doc.toprettyxml(indent = ''))  
    f.close()  

if __name__ == '__main__':  
    class_ind=('van', 'tram', 'car', 'pedestrian', 'truck')#Modify to 5 categories  
    cur_dir=os.getcwd()  
    labels_dir=os.path.join(cur_dir,'label_2')  
    for parent, dirnames, filenames in os.walk(labels_dir): # Get the root directory, subdirectories and files in the root directory respectively     
        for file_name in filenames:  
            full_path=os.path.join(parent, file_name) # Get the full path of the file  
            #print full_path  
            f=open(full_path)  
            split_lines = f.readlines()  
            name= file_name[:-4] # The last four digits are the extension .txt, only the previous file name  
            #print name  
            img_name=name+'.jpg'   
            img_path=os.path.join('JPEGImages폴더경로',img_name) # The path needs to be modified by yourself              
            #print img_path  
            img_size=cv2.imread(img_path).shape  
            generate_xml(name,split_lines,img_size,class_ind)  
print('all txts has converted into xmls')  