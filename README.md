# MAF_GQ_images

Graph mode tensorflow 2.0 MAF implementation (reverts to older behavior).  The tf.dataset has a memory leak, so generally avoiding this one in favor of an [albeit slower] eager implementation in 2.0 which is significantly more memory efficient.
