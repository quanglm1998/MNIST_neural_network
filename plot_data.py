import matplotlib.pyplot as plt


def show_image(arr):
    """Show gray scale image based on arr""" 
    plt.imshow(arr, cmap='gray')
    plt.show()

def show_list(l):
    plt.plot(l)
    plt.show()