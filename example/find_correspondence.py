import cv2
import numpy as np
import sys
import os
import time
sys.path.append(os.path.abspath('.'))
import textrans

if __name__ == '__main__':
    data_dir = os.path.join('data', 'find_correspondence')
    src_obj_path = os.path.join(data_dir, "spot_triangulated.obj")
    src_tex_path = os.path.join(data_dir, "spot_texture.png")
    dst_obj_path = os.path.join(data_dir, "spot_remesh.obj")
    dst_tex_path = os.path.join(data_dir, "spot_remesh_texture")
    src_verts, src_uvs, _, src_vertIDs, src_uvIDs, _, _ = textrans.loadObj(
        src_obj_path)
    dst_verts, dst_uvs, _, dst_vertIDs, dst_uvIDs, _, _ = textrans.loadObj(
        dst_obj_path)
    src_tex = cv2.imread(src_tex_path)

    # Spot uv is not in [0, 1]
    # So simply wrap back
    updated_src_uvs = []
    for uv in src_uvs:
        if uv[0] < 0:
            uv[0] = 1.0 + uv[0]
        if 1 < uv[0]:
            uv[0] = uv[0] - 1.0
        updated_src_uvs.append(uv)

    src_uvs = updated_src_uvs
    # Copy and paste texture colors to dst uv space
    start_t = time.time()
    dst_tex, dst_mask, nn_fid_tex, nn_pos_tex, nn_bary_tex = textrans.transferWithoutCorrespondence(
        src_uvs, src_uvIDs, src_verts,
        src_vertIDs, src_tex,
        dst_uvs, dst_uvIDs, dst_verts,
        dst_vertIDs,
        1024, 1024)
    end_t = time.time()
    print("transferWithoutCorrespondence",
          '{:.2f}'.format(end_t - start_t), "sec")
    # To erase bleeding on boundaries of uv,
    # apply inpaint (alternately, color dilation would also work)
    start_t = time.time()
    inpaint_mask = np.bitwise_not(dst_mask)
    inpaint_mask = cv2.dilate(
        inpaint_mask, np.ones((3, 3), np.uint8), iterations=1)
    dst_tex_inpainted = cv2.inpaint(
        dst_tex, inpaint_mask, 3, cv2.INPAINT_TELEA)
    print("inpaint", '{:.2f}'.format(end_t - start_t), "sec")
    cv2.imwrite(dst_tex_path + ".png", dst_tex_inpainted)
    cv2.imwrite(dst_tex_path + "_org.png", dst_tex)
    cv2.imwrite(dst_tex_path + "_mask.png", dst_mask)
    nn_fid_tex_vis = (nn_fid_tex / len(src_vertIDs) * 255).astype(np.uint8)
    cv2.imwrite(dst_tex_path + "_fid.png", nn_fid_tex_vis)
    pos_max = np.max(nn_pos_tex, axis=(0, 1))
    pos_min = np.amin(nn_pos_tex, axis=(0, 1))
    nn_pos_tex_vis = ((nn_pos_tex - pos_min) /
                      (pos_max - pos_min) * 255).astype(np.uint8)
    cv2.imwrite(dst_tex_path + "_pos.png", nn_pos_tex_vis)
    nn_bary_tex_vis = (nn_bary_tex * 255).astype(np.uint8)
    nn_bary_tex_vis[..., 2] = 0
    cv2.imwrite(dst_tex_path + "_bary.png", nn_bary_tex_vis)
