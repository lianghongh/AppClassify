import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from matplotlib import pyplot as plt


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

    return np.array(data),np.array(tag)

def best_k(X,y):
    size=int(np.sqrt(len(X)))
    k_range=range(1,size+1)
    k_scores=[]
    for k in k_range:
        knn=KNeighborsClassifier(n_neighbors=k)
        scores=cross_val_score(knn,X,y,scoring="accuracy",cv=4)
        k_scores.append(scores.mean())

    plt.plot(k_range,k_scores,marker='o')
    plt.title("K value for knn model")
    plt.xlabel("k value for knn")
    plt.ylabel("Cross Validation Accuracy")
    plt.legend()
    plt.show()

    k_v=sorted(zip(k_scores,k_range),key=lambda x:x[0],reverse=True)
    return k_v[0][1]

if __name__=='__main__':
    group,label=readData("/Users/lianghong/Downloads/gem5_result")
    k=best_k(group,label)
    print("best k=%d" %k)
    knn=KNeighborsClassifier(k)
    knn=knn.fit(group,label)

    print("分类准确率：%f" % knn.score(group,label))

