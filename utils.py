import sys
from numba import jit, prange


epsilon = sys.float_info.epsilon


@jit(nopython=True, cache=True, parallel=True)
def pt_in_line_w_tol(pt1, pt2, pt3, tol=10):
    result = 0

    for i in prange(-tol, tol + 1):
        for j in prange(-tol, tol + 1):
            if pt_in_line(pt1, pt2, (pt3[0] + i, pt3[1] + j)):
                result += 1

    return result != 0


@jit(nopython=True, cache=True, fastmath=True)
def pt_in_line(pt1, pt2, pt3):
    crossproduct = (pt3[1] - pt1[1]) * (pt2[0] - pt1[0]) - (pt3[0] - pt1[0]) * (
        pt2[1] - pt1[1]
    )
    if abs(crossproduct) > epsilon:
        return False

    dotproduct = (pt3[0] - pt1[0]) * (pt2[0] - pt1[0]) + (pt3[1] - pt1[1]) * (
        pt2[1] - pt1[1]
    )
    if dotproduct < 0:
        return False

    squaredlengthba = (pt2[0] - pt1[0]) * (pt2[0] - pt1[0]) + (pt2[1] - pt1[1]) * (
        pt2[1] - pt1[1]
    )
    if dotproduct > squaredlengthba:
        return False

    return True
