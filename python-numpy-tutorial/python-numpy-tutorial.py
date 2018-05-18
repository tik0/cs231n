#http://cs231n.github.io/python-numpy-tutorial/

#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))
plt.show() 

#%%
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    print(pivot,arr,left+middle+right)
    return quicksort(left) + middle + quicksort(right)
print(quicksort([3,6,8,10,1,2,1]))

#%%
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0,2])
y = x + v  # Add v to each row of x using broadcasting
print(y)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"

#%%
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
y= x-5

#%%
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.plot(x, y)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('xx')
plt.legend(['Sine','Cosine','Line'])
plt.show()

#%%
plt.subplot(1,2,1)
plt.plot(x, y_sin)
plt.title('Sine')

plt.subplot(1,2,2)
plt.plot(x, y_cos)
plt.title('Cosine')

plt.show