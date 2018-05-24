import matplotlib.pyplot as plt


def show_image(arr):
    """Show gray scale image based on arr""" 
    plt.imshow(arr, cmap='gray')
    plt.show()

def show_list(l):
    plt.plot(l)
    plt.show()

if __name__ == "__main__":
    fin = open("log.txt", "r")
    cost = []
    cost_cv = []
    iter = 0
    for line in fin:
        a = line[:-1].split(" ")
        cost.append(float(a[1]))
        cost_cv.append(float(a[2]))
    plt.plot(cost, color="red")
    plt.plot(cost_cv, color="blue")
    plt.show()