import tensorflow as tf
import numpy as np

def readData(path):
    data=[]
    tag_d={"SF":0,"SS":1,"IF":2,"IS":3,"IR":4}
    benchmark=["bwaves","gamess","gromacs","hmmer","leslie3d","mcf","sjeng","astar","bzip2","calculix","gobmk","h264ref","lbm","libquantum","milc","namd","soplex","tonto","zeusmp","GemsFDTD","omnetpp"]
    for ff in benchmark:
        file=path+"/"+ff+"/miss_rate"
        miss_rate=[]
        with open(file,encoding="utf-8") as f:
            line=f.readline()
            while line!="":
                if line!="\n":
                    miss_rate.append(float(line))
                line=f.readline()
        data.append(miss_rate)
    tag=[tag_d["SS"],tag_d["SF"],tag_d["IF"],tag_d["IF"],tag_d["IS"],tag_d["IF"],tag_d["IS"],tag_d["IR"],tag_d["IF"],tag_d["IS"],tag_d["IR"],tag_d["SS"],tag_d["IR"],tag_d["IR"],tag_d["IR"],tag_d["IS"],tag_d["IF"],tag_d["SF"],tag_d["IS"],tag_d["IS"],tag_d["SF"]]
    ll=len(benchmark)
    label=[]
    for i in range(ll):
        clazz=[0 for j in range(len(tag_d))]
        clazz[tag[i]]=1
        label.append(clazz)

    return np.array(data),np.array(label)


def Softmax_Regression(path):
    data,label=readData(path)
    INPUT_NODE,OUTPUT_NODE=11,5
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x')
    W=tf.Variable(tf.zeros([INPUT_NODE,OUTPUT_NODE]))
    b=tf.Variable(tf.zeros([OUTPUT_NODE]))
    y=tf.nn.softmax(tf.matmul(x,W)+b)

    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE])
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    train_step=tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        size=len(data)
        for i in range(size):
            xs,ys=data[i:i+1],label[i:i+1]
            train_step.run({x:xs,y_:ys})

        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        print(acc.eval({x:data,y_:label}))


if __name__=="__main__":
    Softmax_Regression("/Users/lianghong/Downloads/gem5_result")