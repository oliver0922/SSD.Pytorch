#!/usr/bin/env python  
# -*- coding: UTF-8 -*-  

# modify_annotations_txt.py  
import glob  
import string  

txt_list = glob.glob('/content/drive/MyDrive/VOC2007/label_2/*.txt') # Store all txt file paths in the Labels folder  
def show_category(txt_list):  
    category_list= []  
    for item in txt_list:  
        try:  
            with open(item) as tdf:  
                for each_line in tdf:  
                    labeldata = each_line.strip().split(' ') # Remove the extra characters before and after and separate them  
                    category_list.append(labeldata[0]) # As long as the first field is the category  
        except IOError as ioerr:  
            print('File error:'+str(ioerr))  
    print(set(category_list)) # Output collection  

def merge(line):  
    each_line=''  
    for i in range(len(line)):  
        if i!= (len(line)-1):  
            each_line=each_line+line[i]+' '  
        else:  
            each_line=each_line+line[i] # No space after the last field  
    each_line=each_line+'\n'  
    return (each_line)  

print('before modify categories are:\n')  
show_category(txt_list)  

for item in txt_list:  
    new_txt=[]  
    try:  
        with open(item, 'r') as r_tdf:  
            for each_line in r_tdf:  
                labeldata = each_line.strip().split(' ')

                '''if labeldata[0] in ['Truck','Van','Tram','Car']: # merge car  
                    labeldata[0] = labeldata[0].replace(labeldata[0],'car')  
                                 if labeldata[0] in ['Person_sitting','Cyclist','Pedestrian']: # merge line human  
                    labeldata[0] = labeldata[0].replace(labeldata[0],'pedestrian')'''
                #print type(labeldata[4])
                if labeldata[4] == '0.00':
                    labeldata[4] = labeldata[4].replace(labeldata[4],'1.00')
                if labeldata[5] == '0.00':
                    labeldata[5] = labeldata[5].replace(labeldata[5],'1.00')
                if labeldata[0] == 'Truck':  
                    labeldata[0] = labeldata[0].replace(labeldata[0],'truck')
                if labeldata[0] == 'Van':  
                    labeldata[0] = labeldata[0].replace(labeldata[0],'van') 
                if labeldata[0] == 'Tram':  
                    labeldata[0] = labeldata[0].replace(labeldata[0],'tram')
                if labeldata[0] == 'Car':  
                    labeldata[0] = labeldata[0].replace(labeldata[0],'car')
                #if labeldata[0] == 'Cyclist':  
                    #labeldata[0] = labeldata[0].replace(labeldata[0],'cyclist')
                if labeldata[0] in ['Person_sitting','Pedestrian']: # Merging line human  
                    labeldata[0] = labeldata[0].replace(labeldata[0],'pedestrian')
                if labeldata[0] == 'Cyclist':  
                    continue
                if labeldata[0] == 'DontCare': # Ignore the Dontcare class  
                    continue  
                if labeldata[0] == 'Misc': # Ignore Misc class  
                    continue  
                new_txt.append(merge(labeldata)) # Rewrite the new txt file  
        with open(item,'w+') as w_tdf: # w+ is to open the original file and delete the content, and write the new content in  
            for temp in new_txt:  
                w_tdf.write(temp)  
    except IOError as ioerr:  
        print('File error:'+str(ioerr))  

print('\nafter modify categories are:\n')  
show_category(txt_list)