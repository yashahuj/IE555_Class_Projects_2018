# Source(s):  
# [enter the URLs for any Websites that were used for this code.]
# [this includes data sources and code sources.]

# I've pasted some sample code as a placeholder here.

import matplotlib.pyplot as plt

x = [1, 3, 4, 7]
y = [2, 5, 1, 6]

for i in range(0,len(x)):
	print "x[%d] = %f" % (i, x[i])		
	
plt.plot(x, y)
plt.savefig('samplefigure.png')	
plt.show()