from reid.lsh import LSHash
import numpy as np
import math

def find_closest_index(query, gallery, top_k=1):
    dim = gallery.shape[1]
    N = gallery.shape[0]
    lsh = LSHash(max(2, int(math.log(N))), dim)
    for point in gallery:
        lsh.index(point)
    result = lsh.query(query, top_k)
    return [np.where((gallery == np.array(r[0])).all(1))[0][0] for r in result]

if __name__=="__main__":
    query = np.random.randint(1, high=100, size=(256))
    gallery = np.random.randint(1, high=100, size=(300,256))
    # print(query)
    # print(gallery)
    print(len(find_closest_index(query, gallery, top_k=154)))