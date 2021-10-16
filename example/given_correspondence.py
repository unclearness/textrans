import cv2
import numpy as np
import sys
import os
import time
sys.path.append(os.path.abspath('.'))
import uvtransfer

if __name__ == '__main__':
    data_dir = os.path.join('data', 'given_correspondence')
    src_obj_path = os.path.join(data_dir, "bunny_textured_charts.obj")
    src_tex_path = os.path.join(data_dir, "bunny_textured_charts.png")
    dst_obj_path = os.path.join(data_dir, "bunny.obj")
    dst_tex_path = os.path.join(data_dir, "bunny.png")
    _, src_uvs, _, _, src_uvIDs, _, _ = uvtransfer.loadObj(src_obj_path)
    _, dst_uvs, _, _, dst_uvIDs, _, _ = uvtransfer.loadObj(dst_obj_path)
    src_tex = cv2.imread(src_tex_path)
    # Copy and paste texture colors to dst uv space
    dst_tex, dst_mask = uvtransfer.transfer(src_uvs, src_uvIDs, src_tex, dst_uvs,
                                            dst_uvIDs,
                                            512, 512, super_sample=2.0)
    # To erase bleeding on boundaries of uv,
    # apply inpaint (alternately, color dilation would also work)
    inpaint_mask = np.bitwise_not(dst_mask)
    inpaint_mask = cv2.dilate(
        inpaint_mask, np.ones((3, 3), np.uint8), iterations=2)
    dst_tex_inpainted = cv2.inpaint(
        dst_tex, inpaint_mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(dst_tex_path, dst_tex_inpainted)
