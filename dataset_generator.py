import tensorflow as tf

class Data():

    def __init__(self,data_dir):
        list_ds = tf.data.Dataset.list_files(data_dir+"/**.png")
        for f in list_ds.take(5):
            print(f.numpy())
        ## ...
