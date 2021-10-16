import numpy as np

# http://www.cloud.teu.ac.jp/public/MDF/toudouhk/blog/2015/01/15/OBJTips/


def loadObj(fliePath):
    numVertices = 0
    numUVs = 0
    numNormals = 0
    numFaces = 0
    vertices = []
    uvs = []
    normals = []
    vertexColors = []
    faceVertIDs = []
    uvIDs = []
    normalIDs = []
    for line in open(fliePath, "r"):
        vals = line.split()
        if len(vals) == 0:
            continue
        if vals[0] == "v":
            v = [float(x) for x in vals[1:4]]
            vertices.append(v)
            if len(vals) == 7:
                vc = [float(x) for x in vals[4:7]]
                vertexColors.append(vc)
            numVertices += 1
        if vals[0] == "vt":
            vt = [float(x) for x in vals[1:3]]
            uvs.append(vt)
            numUVs += 1
        if vals[0] == "vn":
            vn = [float(x) for x in vals[1:4]]
            normals.append(vn)
            numNormals += 1
        if vals[0] == "f":
            fvID = []
            uvID = []
            nvID = []
            for f in vals[1:]:
                w = f.split("/")
                if numVertices > 0:
                    fvID.append(int(w[0])-1)
                if numUVs > 0:
                    uvID.append(int(w[1])-1)
                if numNormals > 0:
                    nvID.append(int(w[2])-1)
            faceVertIDs.append(fvID)
            uvIDs.append(uvID)
            normalIDs.append(nvID)
            numFaces += 1
    return np.array(vertices), np.array(uvs), np.array(normals), \
        faceVertIDs, uvIDs, normalIDs, vertexColors


def saveObj(filePath, vertices, uvs, normals,
            faceVertIDs, uvIDs, normalIDs, vertexColors):
    f_out = open(filePath, 'w')
    f_out.write("####\n")
    f_out.write("#\n")
    f_out.write("# Vertices: %s\n" % (len(vertices)))
    f_out.write("# Faces: %s\n" % (len(faceVertIDs)))
    f_out.write("#\n")
    f_out.write("####\n")
    for vi, v in enumerate(vertices):
        vStr = "v %s %s %s" % (v[0], v[1], v[2])
        if len(vertexColors) > 0:
            color = vertexColors[vi]
            vStr += " %s %s %s" % (color[0], color[1], color[2])
        vStr += "\n"
        f_out.write(vStr)
    f_out.write("# %s vertices\n\n" % (len(vertices)))
    for uv in uvs:
        uvStr = "vt %s %s\n" % (uv[0], uv[1])
        f_out.write(uvStr)
    f_out.write("# %s uvs\n\n" % (len(uvs)))
    for n in normals:
        nStr = "vn %s %s %s\n" % (n[0], n[1], n[2])
        f_out.write(nStr)
    f_out.write("# %s normals\n\n" % (len(normals)))
    for fi, fvID in enumerate(faceVertIDs):
        fStr = "f"
        for fvi, fvIDi in enumerate(fvID):
            fStr += " %s" % (fvIDi + 1)
            if len(uvIDs) > 0:
                fStr += "/%s" % (uvIDs[fi][fvi] + 1)
            if len(normalIDs) > 0:
                fStr += "/%s" % (normalIDs[fi][fvi] + 1)
        fStr += "\n"
        f_out.write(fStr)
    f_out.write("# %s faces\n\n" % (len(faceVertIDs)))
    f_out.write("# End of File\n")
    f_out.close()
