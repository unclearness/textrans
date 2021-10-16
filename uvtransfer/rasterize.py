import numpy as np
import cv2


def U2X(u, w):
    return u * w - 0.5


def V2Y(v, h, flip=True):
    if flip:
        return (1.0 - v) * h - 0.5
    return v * h - 0.5


def X2U(x, w):
    return (x + 0.5) / w


def Y2V(y, h, flip=True):
    v = (y + 0.5) / h
    if flip:
        return 1.0 - v
    return v


def bilinearInterpolation(x, y, tex):
    min_x = int(np.floor(x))
    max_x = min_x + 1
    min_y = int(np.floor(y))
    max_y = min_y + 1
    local_u = x - min_x
    local_v = y - min_y
    interp = (1.0 - local_u) * (1.0 - local_v) * tex[min_y, min_x] \
        + local_u * (1.0 - local_v) * tex[min_y, max_x] \
        + (1.0 - local_u) * local_v * tex[max_y, min_x] \
        + local_u * local_v * tex[max_y, max_x]
    return interp


def edgeFunction(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])


def transfer(src_uvs, src_uv_faces, src_tex, dst_uvs, dst_uv_faces,
             dst_tex_h, dst_tex_w, super_sample=2.0):
    if super_sample < 1.0:
        super_sample = 1.0
    dst_tex_h_ = int(dst_tex_h * super_sample)
    dst_tex_w_ = int(dst_tex_w * super_sample)
    if 2 < len(src_tex.shape):
        src_h, src_w, src_c = src_tex.shape
        dst_tex_shape = (dst_tex_h_, dst_tex_w_, src_c)
    else:
        src_h, src_w = src_tex.shape
        dst_tex_shape = (dst_tex_h_, dst_tex_w_)
    dst_tex_size = (dst_tex_h, dst_tex_w)
    dst_tex = np.zeros(dst_tex_shape, dtype=src_tex.dtype)
    dst_mask = np.zeros((dst_tex_h_, dst_tex_w_), dtype=np.uint8)

    for sface, dface in zip(src_uv_faces, dst_uv_faces):
        # Get bounding box in dst tex
        duv0 = dst_uvs[dface[0]]
        duv1 = dst_uvs[dface[1]]
        duv2 = dst_uvs[dface[2]]
        suv0 = src_uvs[sface[0]]
        suv1 = src_uvs[sface[1]]
        suv2 = src_uvs[sface[2]]
        bb_min_x = U2X(max(0.0, min([duv0[0], duv1[0], duv2[0]])), dst_tex_w_)
        bb_max_x = U2X(
            min(dst_tex_w_-1, max([duv0[0], duv1[0], duv2[0]])), dst_tex_w_)
   
        # Be careful how to get min/max of y. It is an inversion of v
        bb_min_v = min([duv0[1], duv1[1], duv2[1]])
        bb_max_v = max([duv0[1], duv1[1], duv2[1]])
        bb_min_y = max(0, V2Y(bb_max_v, dst_tex_h_))
        bb_max_y = min(dst_tex_h_-1, V2Y(bb_min_v, dst_tex_h_))

        # pixel-wise loop for the bb in dst tex
        area = edgeFunction(duv0, duv1, duv2)
        inv_area = 1.0 / area
        # print(int(bb_min_y), int(bb_max_y), int(bb_min_x), int(bb_max_x))
        for j in range(int(bb_min_y), int(np.ceil(bb_max_y))):
            for i in range(int(bb_min_x), int(np.ceil(bb_max_x))):
                # Calc barycentric coordinate
                pix_uv = (X2U(i, dst_tex_w_), Y2V(j, dst_tex_h_))
                w0 = edgeFunction(duv1, duv2, pix_uv)
                w1 = edgeFunction(duv2, duv0, pix_uv)
                w2 = edgeFunction(duv0, duv1, pix_uv)
                w0 *= inv_area
                w1 *= inv_area
                w2 *= inv_area
                # Check if this pixel is on the dst triangle
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                # Calc UV in src tex through barycentric
                suv = w0 * suv0 + w1 * suv1 + w2 * suv2
                # Calc pixel pos in src tex
                sx = U2X(suv[0], src_w)
                sy = V2Y(suv[1], src_h)

                # Fetch and copy to dst tex
                src_color = bilinearInterpolation(sx, sy, src_tex)
                #src_color = src_tex[int(sy), int(sx)]
                #src_color = np.clip(src_color, 0, 255)
                dst_tex[j, i] = src_color.astype(src_tex.dtype)
                dst_mask[j, i] = 255

    if 1.0 < super_sample:
        dst_tex = cv2.resize(dst_tex, dst_tex_size,
                             interpolation=cv2.INTER_AREA)
        dst_mask = cv2.resize(dst_mask, dst_tex_size,
                             interpolation=cv2.INTER_NEAREST)
    return dst_tex, dst_mask
