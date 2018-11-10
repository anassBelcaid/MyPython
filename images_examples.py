"""
set of image examples
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import circle, polygon
from scipy import ndimage


def four_parts(n=200, values=[0, 0.33, 0.66, 0.99]):
    """
    function to generate the testing image with four parts
    the value of each part is stored in values
    """
    image = np.zeros((n, n), dtype=np.float32)
    t1 = int(n/3)              # one third of the image
    t3 = int(2*n/3)            # two third of the image

    # the firt upper left part
    Tx = [0, 0, t1, t1]
    Ty = [0, t1, t3, 0]
    Tx, Ty = polygon(Tx, Ty)
    image[Tx, Ty] = values[0]

    # the square in the bottom ouest with value v3
    image[t1:, :t3] = values[2]

    # drawing the upper east triangle
    Tx = [0, 0, t3]
    Ty = [t1, n, n]
    Tx, Ty = polygon(Tx, Ty)
    image[Tx, Ty] = values[1]

    # Last south right part
    Tx = [t1, t3, n, n]
    Ty = [t3, n, n, t3]
    Tx, Ty = polygon(Tx, Ty)
    image[Tx, Ty] = values[3]

    return image


def two_rectangles(n=200, pos=0.5, vertical=True):
    """
    two rectangles images
    """

    image = np.zeros((n, n), dtype=np.float32)
    pos = int(n*pos)
    if(vertical):
        image[:, pos:] = 1.0
    else:
        image[pos:, :] = 1.0
    return image


def circle_inside(n=200, radius=0.2):
    """
    draw a circle inside
    """
    image = np.zeros((n, n), dtype=np.float32)

    # radius scaling
    radius *= n
    # print(radius)

    # getting the circle
    rc, cc = circle(n/2, n/2, radius)
    image[rc, cc] = 1.0

    return image


def shapes(n=200):
    """
    all the classical shapes
    """

    image = np.zeros((n, n), dtype=np.float32)

    # first rectangle
    n_8 = int(n/8)
    n_2 = int(n/2)
    n_4 = int(n/4)
    image[n_8:3*n_8, n_8:3*n_8] = 1.0

    # circle
    cr, cc = circle(n_4, 3*n_4, n_8)
    image[cr, cc] = 1.0

    # polygon
    poly_x = [n_2, 3*n_4, n, 3*n_4]
    poly_y = [n_2, 3*n_4, n_2, n_4]
    pr, pc = polygon(poly_x, poly_y)
    image[pr, pc] = 1.0

    return image


def scipy_segmentation_example(n=200, parts=5):
    """
    example from the scipy lecture
    """
    np.random.seed(1)

    im = np.zeros((n, n))
    points = n*np.random.random((2, parts**2))
    im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    im = ndimage.gaussian_filter(im, sigma=n/(4.*parts))

    mask = (im > im.mean()).astype(np.float)

    mask += 0.1 * im

    img = mask

    return img


if __name__ == '__main__':

    # mat=two_rectangles(vertical=False);
    # mat=circle_inside()
    # mat=shapes()
    # mat = scipy_segmentation_example()
    mat = four_parts(100, [1, 0.2, 0.3, 0.8])
    print(mat.shape)
    plt.imshow(mat, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()
