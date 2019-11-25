# MAF_GQ_images

Graph mode tensorflow 2.0 MAF implementation (reverts to older behavior).  The tf.dataset in graph mode seems to have a memory leak that could be fixed by avoiding tf.datasets altogether.  Just change input pipeline to use non-tf functions -- although pipelining the preprocessing functions in parallel isn't nearly as clean.  So generally best to avoid this one in favor of an [albeit slightly slower training loop] eager implementation in 2.0, which is significantly more memory efficient.
