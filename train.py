import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import numpy.random as rng
import os

class idx_streamer:
    """
    Index streaming class to get randomized batch size indices samples in a uniform way (using epochs).
    """
    def __init__(self,N,batch_size):
        """
        Constructor defining streamer parameters.
        :param N: total number of samples.
        :param batch_size: batch size that the streamer has to generate.
        """
        self.N = N
        self.sequence = np.arange(N)
        self.batch_size = batch_size
        self.stream = []
        self.epoch = -1
        self.vh = 0
    
    def gen(self):
        """
        Index stream generation function. Outputs next batch indices.
        :return: List of batch indices.
        """
        while len(self.stream) < self.batch_size:
            rng.shuffle(self.sequence)
            self.stream += list(self.sequence)
            self.epoch +=1
        stream = self.stream[:self.batch_size]
        self.stream = self.stream[self.batch_size:]
        return stream

class Trainer:
    """
    Training class for the standard MADEs/MAFs classes using a tensorflow optimizer.
    """
    def __init__(self, model, optimizer=tf.train.AdamOptimizer, optimizer_arguments={}, model_contrastive=None, negative=False):
        """
        Constructor that defines the training operation.
        :param model: made/maf instance to be trained.
        :param optimizer: tensorflow optimizer class to be used during training.
        :param optimizer_arguments: dictionary of arguments for optimizer intialization.
        """
        
        self.model = model
        self.model_contrastive = model_contrastive

        if hasattr(self.model,'batch_norm') and self.model.batch_norm is True:
            self.has_batch_norm = True
        else:
            self.has_batch_norm = False
        self.train_op = optimizer(**optimizer_arguments).minimize(self.model.trn_loss)
        # self.train_op = optimizer(**optimizer_arguments).minimize(-tfp.stats.percentile(self.model.L, 10))

    def train(self, sess, train_data_in, val_data_in, test_data_in, max_iterations=1000,
              early_stopping=20, check_every_N=5, saver_name='tmp_model', show_log=False):
        """
        Training function to be called with desired parameters within a tensorflow session.
        :param sess: tensorflow session where the graph is run.
        :param train_data: train data to be used.
        :param val_data: validation data to be used for early stopping. If None, train_data is splitted 
             into p_val percent for validation randomly.  
        :param p_val: percentage of training data randomly selected to be used for validation if
             val_data is None.
        :param max_iterations: maximum number of iterations for training.
        :param batch_size: batch size in each training iteration.
        :param early_stopping: number of iterations for early stopping criteria.
        :param check_every_N: check every N iterations if model has improved and saves if so.
        :param saver_name: string of name (with or without folder) where model is saved. If none is given,
            a temporal model is used to save and restore best model, and removed afterwards.
        """
        # Early stopping variables
        bst_loss = np.infty
        early_stopping_count = 0
        saver = tf.train.Saver()

        # Main training loop
        for iteration in range(max_iterations):
            train_data = sess.run(train_data_in)
            if self.has_batch_norm:
                sess.run(self.train_op,feed_dict={self.model.input:train_data,self.model.training:True})
            else:
                sess.run(self.train_op,feed_dict={self.model.input:train_data})
            # Early stopping check
            if iteration%check_every_N == 0:
                val_data = sess.run(val_data_in)
                if self.has_batch_norm:
                    self.model.update_batch_norm(train_data,sess)
                # this_loss = sess.run(self.model.trn_loss,feed_dict={self.model.input:val_data})
                this_loss = -np.ma.masked_invalid(sess.run(self.model.L, feed_dict={self.model.input: val_data})).mean()

                if show_log:
                    that_loss = 0
                    test_data = sess.run(test_data_in)
                    train_loss = sess.run(self.model.trn_loss,feed_dict={self.model.input:train_data})
                    if test_data is not None and type(test_data) == np.ndarray:
                        # that_loss = sess.run(self.model.trn_loss, feed_dict={self.model.input: test_data})
                        that_loss = -np.ma.masked_invalid(sess.run(self.model.L, feed_dict={self.model.input: test_data})).mean()

                    print("Iteration {:05d}, Train_loss: {:05.4f}, Val_loss: {:05.4f}, , Test_loss: {:05.4f}".format(iteration,train_loss,this_loss, that_loss))
                if this_loss < bst_loss:
                    bst_loss = this_loss
                    # saver.save(sess,"./"+saver_name)
                    model_parms = sess.run(self.model.parms)
                    early_stopping_count = 0
                else:
                    early_stopping_count += check_every_N
            if early_stopping_count >= early_stopping:
                break
                
        if show_log:
            print("Training finished")
            print("Best Iteration {:05d}, Val_loss: {:05.4f}".format(iteration-early_stopping,bst_loss))
        # Restore best model and save batch norm mean and variance if necessary
        # saver.restore(sess,"./"+saver_name)
        for m, n in zip(self.model.parms, model_parms):
            sess.run(tf.assign(m, n))

        if self.has_batch_norm:
            self.model.update_batch_norm(train_data,sess)
        
        # Remove model data if temporal model data was used
        if saver_name == 'tmp_model':
            for file in os.listdir("./"):
                if file[:len(saver_name)] == saver_name:
                    os.remove(file)