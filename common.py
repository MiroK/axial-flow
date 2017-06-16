import numpy as np

def axi_mesh(inner_curve, outer_curve, use_spline):
    '''
    DIXME
    '''
    if use_spline:
        return spline_axi_mesh(inner_curve, outer_curve)
    else:
        return linear_axi_mesh(inner_curve, outer_curve)

def linear_axi_mesh(inner_curve, outer_curve):
    points_i, sizes_i = curve_data(inner_curve)
    points_o, sizes_o = curve_data(outer_curve)

    assert np.linalg.norm(points_i[0]
    assert abs(points_i[0][1]-points_o[0][1]) < 1E-13
