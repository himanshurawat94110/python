#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:24:11 2020

@author: himanshurawat
"""
##  Q 1 - for prime numbers 

num = 11


if num>1:
    for i in range(2,num//2):
        if (num%i)==0:
            print(num,"num is not a prime")
            break
        else:
            print(num,"number is a prime")
else:
    print(num,"is not a prime number")





def prime(num):
    
    if num>1:
        for i in range(2,num//2):
            if (num%i)==0:
                print(num,"num is not a prime")
                break
        else:
            print(num,"number is a prime")
    else :
        print(num,"is not a prime number")


prime(7)
prime(978)

# optimised soltuion




num =11 

def isprime(n):
    
    if n <= 1:
        return False
    if n <=3:
        return True
        
    if(n%2 == 0 or n%3==0):
        return False
    i=5
    while(i*i<=n):
        if(n%i==0 or n%(i+2)==0):
            return False
    i=i+6
    return True 
    
isprime(19)


df = pd.DataFrame(["STD, City    State",
"33, Kolkata    West Bengal",
"44, Chennai    Tamil Nadu",
"40, Hyderabad    Telengana",
"80, Bangalore    Karnataka"], columns=['row'])

# Solution
df_out = df.row.str.split(',|\t', expand=True)

# Make first row as header
new_header = df_out.iloc[0]
df_out = df_out[1:]
df_out.columns = new_header
print(df_out)

