import textrans
import cv2
import numpy as np
import sys
import os
import time
sys.path.append(os.path.abspath('.'))

if __name__ == '__main__':
    data_dir = os.path.join('data', 'given_correspondence')
    src_obj_path = os.path.join(data_dir, "bunny_textured_charts.obj")
    src_tex_path = os.path.join(data_dir, "bunny_textured_charts.png")
    dst_obj_path = os.path.join(data_dir, "bunny.obj")
    dst_tex_path = os.path.join(data_dir, "bunny.png")
    src_verts, src_uvs, _, src_vertIDs, src_uvIDs, _, _ = textrans.loadObj(
        src_obj_path)
    dst_verts, dst_uvs, _, dst_vertIDs, dst_uvIDs, _, _ = textrans.loadObj(
        dst_obj_path)
    src_tex = cv2.imread(src_tex_path)
    # Copy and paste texture colors to dst uv space
    dst_tex, dst_mask = textrans transferWithoutCorrespondence(
        src_uvs, src_uvIDs, src_verts,
        src_vertIDs, src_tex,
        dst_uvs, dst_uvIDs, dst_verts,
        dst_vertIDs,
        64, 64)
    # To erase bleeding on boundaries of uv,
    # apply inpaint (alternately, color dilation would also work)
    inpaint_mask = np.bitwise_not(dst_mask)
    inpaint_mask = cv2.dilate(
        inpaint_mask, np.ones((3, 3), np.uint8), iterations=2)
    dst_tex_inpainted = cv2.inpaint(
        dst_tex, inpaint_mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(dst_tex_path, dst_tex_inpainted)
