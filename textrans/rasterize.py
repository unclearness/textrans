import numpy as np
import cv2
import scipy.spatial as ss
from textrans import util

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

def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2


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


def triArea(a, b, c):
    return 0.5 * np.linalg.norm(np.cross((b - a), (c - a)))


def transferCommonProcess(src_tex, dst_tex_h, dst_tex_w, super_sample):
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
    return dst_tex, dst_mask, dst_tex_size, dst_tex_h_,\
        dst_tex_w_, src_h, src_w, super_sample


def transfer(src_uvs, src_uv_faces, src_tex, dst_uvs, dst_uv_faces,
             dst_tex_h, dst_tex_w, super_sample=2.0):

    dst_tex, dst_mask, dst_tex_size, dst_tex_h_, dst_tex_w_,\
        src_h, src_w, super_sample = transferCommonProcess(src_tex,
                                                           dst_tex_h,
                                                           dst_tex_w,
                                                           super_sample)

    for sface, dface in zip(src_uv_faces, dst_uv_faces):
        # Get bounding box in dst tex
        duv0 = dst_uvs[dface[0]]
        duv1 = dst_uvs[dface[1]]
        duv2 = dst_uvs[dface[2]]
        suv0 = src_uvs[sface[0]]
        suv1 = src_uvs[sface[1]]
        suv2 = src_uvs[sface[2]]
        bb_min_x = U2X(
            max(0.0, min([duv0[0], duv1[0], duv2[0]])), dst_tex_w_)
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
                # src_color = src_tex[int(sy), int(sx)]
                # src_color = np.clip(src_color, 0, 255)
                dst_tex[j, i] = src_color.astype(src_tex.dtype)
                dst_mask[j, i] = 255

    if 1.0 < super_sample:
        dst_tex = cv2.resize(dst_tex, dst_tex_size,
                             interpolation=cv2.INTER_AREA)
        dst_mask = cv2.resize(dst_mask, dst_tex_size,
                              interpolation=cv2.INTER_NEAREST)
    return dst_tex, dst_mask


def computeFaceInfo(verts, vert_faces):
    face_centroids = []
    # ax+by+cz+d=0
    face_planes = []
    for face in vert_faces:
        v0 = verts[face[0]]
        v1 = verts[face[1]]
        v2 = verts[face[2]]
        centroid = (v0 + v1 + v2) / 3.0
        face_centroids.append(centroid)

        vec10 = v1 - v0
        vec20 = v2 - v0
        n = normalize(np.cross(vec10, vec20))
        d = -(n.dot(v0))
        face_planes.append(np.array([n[0], n[1], n[2], d]))
    return face_centroids, face_planes


def calcClosestSurfaceInfo(tree, dst_pos, src_verts, src_verts_faces,
                           src_face_planes, nn_num):
    # Get the closest src face
    # Roughly get candidates. NN for face center points.
    dists, indices = tree.query(dst_pos, k=nn_num)
    # Check point-plane distance and get the smallest
    min_dist = 99999999999999999.99
    min_signed_dist = None
    min_index = None
    min_bary = None
    for index in indices:
        plane = src_face_planes[index]
        # point-plane distance |ax'+by'+cz'+d|
        signed_dist = dst_pos.dot(plane[:3]) + plane[3]
        dist = np.abs(signed_dist)
        if dist < min_dist:
            foot = - signed_dist * plane[:3] + dst_pos
            foot_dist = foot.dot(plane[:3]) + plane[3]
            if np.abs(foot_dist) > 0.0001:
                print("wrong dist ", foot, foot_dist)
                raise Exception()
            # Calc barycentric of the crossing point
            svface = src_verts_faces[index]
            sv0 = src_verts[svface[0]]
            sv1 = src_verts[svface[1]]
            sv2 = src_verts[svface[2]]
            area = triArea(sv0, sv1, sv2)
            inv_area = 1.0 / area
            w0 = triArea(sv1, sv2, foot) * inv_area
            w1 = triArea(sv2, sv0, foot) * inv_area
            w2 = triArea(sv0, sv1, foot) * inv_area
            if w0 < 0 or w1 < 0 or w2 < 0 or 1 < w0 or 1 < w1 or 1 < w2:
                continue
            if np.abs(w0 + w1 + w2 - 1.0) > 0.1:
                #print(np.abs(w0 + w1 + w2 - 1.0), (w0, w1, w2))
                continue
            min_dist = dist
            min_signed_dist = signed_dist
            min_index = index
            min_bary = (w0, w1, w2)
    #w2 = 1- w0 - w1
    #
    # 
    # print(min_bary)
    return foot, min_signed_dist, min_dist, min_index, min_bary


def transferWithoutCorrespondence(src_uvs, src_uv_faces, src_verts,
                                  src_verts_faces, src_tex,
                                  dst_uvs, dst_uv_faces, dst_verts,
                                  dst_vert_faces,
                                  dst_tex_h, dst_tex_w,
                                  super_sample=1.0, nn_num=10):
    dst_tex, dst_mask, dst_tex_size, dst_tex_h_, dst_tex_w_,\
        src_h, src_w, super_sample = transferCommonProcess(src_tex,
                                                           dst_tex_h,
                                                           dst_tex_w,
                                                           super_sample)
    # Prepare KD Tree for src face centers
    src_face_centroids, src_face_planes = computeFaceInfo(
        src_verts, src_verts_faces)
    util.saveObj("tmp.obj", src_face_centroids, [], [], [], [], [], [])
    tree = ss.KDTree(src_face_centroids)
    nn_fid_tex = np.zeros((dst_tex_h_, dst_tex_w_), dtype=np.int)
    nn_pos_tex = np.zeros((dst_tex_h_, dst_tex_w_, 3), dtype=np.float)
    nn_bary_tex = np.zeros((dst_tex_h_, dst_tex_w_, 3), dtype=np.float)

    for sface, duvface, dvface in zip(src_uv_faces, dst_uv_faces, dst_vert_faces):
        # Get bounding box in dst tex
        duv0 = dst_uvs[duvface[0]]
        duv1 = dst_uvs[duvface[1]]
        duv2 = dst_uvs[duvface[2]]
        dv0 = dst_verts[dvface[0]]
        dv1 = dst_verts[dvface[1]]
        dv2 = dst_verts[dvface[2]]
        bb_min_x = U2X(
            max(0.0, min([duv0[0], duv1[0], duv2[0]])), dst_tex_w_)
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
        for j in range(int(bb_min_y), int(np.ceil(bb_max_y))):
            for i in range(int(bb_min_x), int(np.ceil(bb_max_x))):
                # Calc barycentric coordinate
                pix_uv = (X2U(i, dst_tex_w_), Y2V(j, dst_tex_h_))
                w0 = edgeFunction(duv1, duv2, pix_uv) * inv_area
                w1 = edgeFunction(duv2, duv0, pix_uv) * inv_area
                w2 = edgeFunction(duv0, duv1, pix_uv) * inv_area
                # Check if this pixel is on the dst triangle
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                # Get corresponding position on the dst face
                dpos = w0 * dv0 + w1 * dv1 + w2 * dv2

                # Get the closest src face info
                foot, min_signed_dist, min_dist,\
                    min_index, bary = calcClosestSurfaceInfo(
                        tree, dpos, src_verts, src_verts_faces,
                        src_face_planes, nn_num)
                if min_index is None:
                    continue
                nn_fid_tex[j, i] = min_index
                nn_pos_tex[j, i] = foot
                nn_bary_tex[j, i] = bary
                suvface = src_uv_faces[min_index]
                suv0 = src_uvs[suvface[0]]
                suv1 = src_uvs[suvface[1]]
                suv2 = src_uvs[suvface[2]]

                #print(bary)
                suv = bary[0] * suv0 + bary[1] * suv1 + bary[2] * suv2

                # Calc pixel pos in src tex
                sx = np.clip(U2X(suv[0], src_w), 0, src_w - 1 - 0.001)
                sy = np.clip(V2Y(suv[1], src_h), 0, src_h - 1 - 0.001)
                #print(suv, sx, sy)
                # Fetch and copy to dst tex
                src_color = bilinearInterpolation(sx, sy, src_tex)
                #src_color = src_tex[int(sy), int(sx)]
                # src_color = np.clip(src_color, 0, 255)
                dst_tex[j, i] = src_color.astype(src_tex.dtype)
                dst_mask[j, i] = 255
    if 1.0 < super_sample:
        dst_tex = cv2.resize(dst_tex, dst_tex_size,
                             interpolation=cv2.INTER_AREA)
        dst_mask = cv2.resize(dst_mask, dst_tex_size,
                              interpolation=cv2.INTER_NEAREST)
        nn_fid_tex = cv2.resize(nn_fid_tex, dst_tex_size,
                              interpolation=cv2.INTER_NEAREST)
        nn_pos_tex = cv2.resize(nn_pos_tex, dst_tex_size,
                              interpolation=cv2.INTER_AREA)
        nn_bary_tex = cv2.resize(nn_bary_tex, dst_tex_size,
                              interpolation=cv2.INTER_AREA)
    return dst_tex, dst_mask, nn_fid_tex, nn_pos_tex, nn_bary_tex

