
import pandas as pd
import re


def Map_to_poly(s):
    
    po=[]
    poly=[]
    first_point=[]   
    Last_point=[]
    l=len(s)-1
    my_list=[]
    st=''
    
    for indx, values in s.items():
        st = values
    
    st=re.sub("['M']","",st)
    st=re.sub("['L']","",st)
    st=re.sub("['Q']","",st)
    st=re.sub("[']']","",st)
    st=re.sub("['[']","",st)
    st=re.sub("[']]']","",st)
    st=re.sub("['],']","",st)
    st=st.split('],')  
    my_list=st 
    
    lll=[]
    lo= len(my_list)-1
    last_st= my_list[-1]
    my_list[-1]=last_st.replace(']]','')
    for i in my_list:
      clean_list = i.split(',')
      for c in clean_list:
         if c.isnumeric and c!='' and c!=' ':
            ta= float(c)
            ta=int(ta)
            lll.append(ta)
            
   
    po=[]
    i=0
 
    while i< len(lll):
       x= lll[i]
       y= lll[i+1]
       tem=[x,y]
       po.append(tem)
       i+=2 
    
    return po
              
    