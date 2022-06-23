import numpy as np
import cv2


def rgb2grayscale(rgb):
    return rgb[:, :, 0] * 0.2989 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114

class DenseToSparse:
    def __init__(self):
        pass

    def dense_to_sparse(self, rgb, depth):
        pass

    def __repr__(self):
        pass

class UniformSampling(DenseToSparse):
    name = "uar"
    def __init__(self, num_samples, max_depth=np.inf):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth

    def __repr__(self):
        return "%s{ns=%d,md=%f}" % (self.name, self.num_samples, self.max_depth)

    def dense_to_sparse(self, rgb, depth):
        """
        Samples pixels with `num_samples`/#pixels probability in `depth`.
        Only pixels with a maximum depth of `max_depth` are considered.
        If no `max_depth` is given, samples in all pixels
        """
        mask_keep = depth > 0
        if self.max_depth is not np.inf:
            mask_keep = np.bitwise_and(mask_keep, depth <= self.max_depth)
        n_keep = np.count_nonzero(mask_keep)
        if n_keep == 0:
            return mask_keep
        else:
            prob = float(self.num_samples) / n_keep
            return np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)

class ORBSampling(DenseToSparse):
    name = "orb_sampler"

    def __init__(self, num_samples, max_depth=np.inf):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.orb_extractor = cv2.ORB_create(nfeatures = 5000, scaleFactor=1.2, fastThreshold= 14, nlevels=8, edgeThreshold = 0)
        self.orb_extractor_r = cv2.ORB_create(nfeatures = 5000, scaleFactor=1.1, fastThreshold= 7, nlevels=8, edgeThreshold = 0)
        self.uniform_sample = UniformSampling(num_samples // 2, max_depth)

    def __repr__(self):
        return "%s{ns=%d,md=%f,dil=%d.%d}" % \
               (self.name, self.num_samples, self.max_depth)

    def dense_to_sparse(self, rgb, depth):
        kp = self.orb_extractor.detect((rgb*255).astype("uint8"),None)

        if len(kp) < self.num_samples:
          #print(f"Number of kp found: {len(kp)}")
          kp = self.orb_extractor_r.detect((rgb*255).astype("uint8"),None)
          #print(f"After: {len(kp)}")

        if len(kp) < self.num_samples:
            kp_s = kp
            self.uniform_sample.num_samples = self.num_samples - len(kp)
            mask = self.uniform_sample.dense_to_sparse(rgb,depth)
        else:
            kp_s = random.sample(kp, self.num_samples)
            mask = np.full(depth.shape, False)
            
        for kp in kp_s:
            mask[int(kp.pt[1]),int(kp.pt[0])] = True
          
        return mask
