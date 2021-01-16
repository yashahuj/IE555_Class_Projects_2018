# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 11:39:12 2018

@author: MohitBaffna
"""

import matplotlib.pyplot as plt
import numpy as np

#!/usr/bin/env python

# make sure to install these packages before running:
# pip install pandas
# pip install sodapy
import sodapy 
import pandas as pd
from sodapy import Socrata

# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
client = Socrata("data.cityofnewyork.us", None)

# Example authenticated client (needed for non-public datasets):
# client = Socrata(data.cityofnewyork.us,
#                  MyAppToken,
#                  userame="user@example.com",
#                  password="AFakePassword")

# First 2000 results, returned as JSON from API / converted to Python list of
# dictionaries by sodapy.
results = client.get("mhu7-c3xb", limit=1000)


results_df = pd.DataFrame.from_records(results)
print results_df
df = results_df.iloc[:,[0,9]]
mylp =  df["alarm_box_borough"].tolist()
address = [ str(x) for x in mylp ]
my_list = df["dispatch_response_seconds_qy"].tolist()
numbers = [ int(x) for x in my_list ]
M=0
timeM= 0
for i in range (1,1000):
    if (address[i] == "MANHATTAN"):
        M = M+ 1
        float(M)
        timeM= timeM+numbers[i]
print "No of Cases in Manhattan = %f " %(M)
avgm= timeM/M
print "The average time to dispatch is = %f sec" %(avgm)
B=0
timeB= 0
for i in range (1,1000):
    if (address[i] == "BRONX"):
        B=B+1
        float(B)
        timeB= timeB+numbers[i]
print "No of Cases in Bronx = %f " %(B)
avgB= timeB/B
print "The average time to dispatch is = %f sec" %(avgB)

BY=0
timeBY= 0
for i in range (1,1000):
    if (address[i] == "BROOKLYN"):
        BY=BY+1
        float(BY)
        timeBY= timeBY+numbers[i]
print "No of Cases in Brooklyn = %f " %(BY)
avgBY= timeBY/BY
print "The average time to dispatch is = %f sec" %(avgBY)
Q=0
timeQ= 0
for i in range (1,1000):
    if (address[i] == "QUEENS"):
        Q=Q+1
        float(Q)
        timeQ= timeQ+numbers[i]
print "No of Cases in Queens = %f " %(Q)
avgQ= timeQ/Q
print "The average time to dispatch is = %f sec" %(avgQ)
S=0
timeS= 0
for i in range (1,1000):
    if (address[i] == "RICHMOND / STATEN ISLAND"):
        S=S+1
        float(S)
        timeS= timeS + numbers[i]
print "No of Cases in Staten Island = %f " %(S)
avgS= timeS/S
print "The average time to dispatch is = %f sec" %(avgS)
M=float(M)
labels = 'Manhattan', 'Queens', 'Staten Island', 'Bronx' ,'Brooklyn'
sizes = []
sizes.append(M)
sizes.append(Q)
sizes.append(S)
sizes.append(B)
sizes.append(BY) 
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','red']
explode = (0.1, 0.1, 0.1, 0.1, 0.1) 
plt.title('% of cases in each borough')
plt.pie(sizes, explode=explode ,labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=0)

plt.axis('equal')
plt.savefig('Piechart.png')
plt.show()


avgtime = (avgm, avgQ, avgS, avgB, avgBY)
avgtimes= float((avgm+avgQ+avgS+avgB+avgBY)/len(sizes))
#print avgtimes

ind = np.arange(len(avgtime)) 
width = 0.35  

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, avgtime, width, color='rebeccapurple', label='Boroughs of NYC')
ax.set_xlabel('Borough')
ax.set_ylabel('Time in Sec')
ax.set_title('Avg time to Dispatch')
ax.axhline(avgtimes, color='grey', linewidth=2)
ax.set_xticks(ind -0.2)
ax.set_xticklabels(('MANHATTAN', 'QUEENS', 'STATEN ISLAND', 'BRONX', 'BROOKLYN'))
ax.legend()
plt.savefig('Bargraph.png')
plt.show()