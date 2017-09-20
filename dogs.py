##Learning how selection of a feature matters
import numpy as np 
import matplotlib.pyplot as plt 

greyhounds = 500
labs = 500

#greyhound's height would be between 24 and 32
greyhound_height = 28 + 4 * np.random.randn(greyhounds)

#lab's height would be between 20 and 28
lab_height = 24 + 4 * np.random.randn(labs)
# print(greyhound_height)
# print(lab_height)

# red color for greyhounds and blue for labs 
plt.hist([greyhound_height, lab_height], stacked = True, color = ['r','b'])
plt.show() 