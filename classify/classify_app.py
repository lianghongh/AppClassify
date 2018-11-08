import numpy as np
from matplotlib import pyplot as plt


def read(fp):
    r=[]
    with open(fp,encoding="utf-8") as f:
        line=f.readline()
        while line!= '':
            if line!='\n':
                r.append(float(line))
            line=f.readline()
    return r


def draw_miss_rate(classify,datapath,savepath):
    name = classify.keys()
    for i in name:
        file = datapath + "/" + i + "/miss_rate"
        x = np.arange(1, 12)
        y = read(file)
        cx = np.arange(1, 11)
        cy = []
        for k in range(10):
            cy.append((y[k] - y[k + 1]) / (128 * (2 ** (k + 1) - 2 ** k) / 1024))
        ff = plt.figure()
        pic1 = ff.add_subplot(2, 1, 1)
        pic1.plot(x, y, marker='o', label="Miss Rate")
        pic1.set_title(i + " Cache miss rate")
        pic1.set_ylabel("Miss Rate")
        pic1.grid()
        pic1.legend()

        pic2 = ff.add_subplot(2, 1, 2)

        pic2.plot(cx, cy, marker='o', label="Cache Capacity Gain")
        # pic2.set_title(i + "Miss change rate")
        pic2.set_ylabel("Miss Rate per MB")
        pic2.grid()
        pic2.legend()

        ff.savefig(savepath + "/" + i,dpi=300)
        plt.close()
        count, cc = 0, 0
        for ss in cy:
            if ss != 0:
                count += 1
                cc += ss

        if count != 0:
            print("The %s average capacity gain is %f" % (i, cc / count))
        else:
            print("The %s average capacity gain is 0.0" % (i))


def classify_app(miss_rate,pth,ps,cth):
    """
    IS:密集流型
    IF:密集友好型
    SS:敏感流型
    SF:敏感友好型
    """
    cy=[]
    for k in range(10):
        cy.append((miss_rate[k] - miss_rate[k + 1]) / (128 * (2 ** (k + 1) - 2 ** k) / 1024))
    count,cc=0,0
    for ss in cy:
        if ss != 0:
            count += 1
            cc += ss
    if count!=0:
        pg=cc/count
    else:
        pg=0

    if pg>pth:
        for p in cy:
            if p<0:
                return "SS"
        for c in range(cth+1,10):
            if cy[c]<=ps:
                return "SF"
        return "SS"
    else:
        for p in cy:
            if p<0:
                return "IS"
        for c in range(cth+1,10):
            if cy[c]<=ps:
                return "IF"
        return "IS"


def cal_acc(pred,real):
    size=len(pred)
    if size!=len(real):
        print("Dict size not equal!")
        exit(1)

    acc_count=0
    for i in pred.keys():
        if pred[i]==real[i]:
            acc_count+=1
    return acc_count/size


def find_best_para(miss_data,real):
    """
    pth_x=[0,1]
    ps_x=[0,1]
    cth=[6,11]

    """
    best_pth,best_ps=0,0
    best_acc=0

    i=0
    while i<1:
        j=0
        while j<1:
             pred=do_classify(miss_data,real_class,i,j,8)
             acc=cal_acc(pred,real)
             if acc>best_acc:
                 best_acc=acc
                 best_pth,best_ps=i,j
             j+=0.01
        i+=0.01

    print("Best accuarcy is %f" %(best_acc))
    print("pth=%f,ps=%f" % (best_pth,best_ps))


def do_classify(miss_data,real_class,pth,ps,cth):
    clz=dict()
    for i in real_class.keys():
        t=classify_app(miss_data[i],pth,ps,cth)
        clz[i]=t

    return clz


if __name__=="__main__":
    data_path="/home/lianghong/Desktop/GraduateData/research3/gem5_result"
    real_class = {"bwaves":"IS", "gamess":"SF", "gromacs":"SF", "hmmer":"SF", "leslie3d":"IS", "mcf":"IS", "sjeng":"IS", "astar":"IS", "bzip2":"SF", "calculix":"IS", "gobmk":"IF",
            "h264ref":"SS", "lbm":"SS", "libquantum":"IS", "milc":"IS", "namd":"SS", "soplex":"SS", "tonto":"SS", "zeusmp":"IS", "omnetpp":"SF", "GemsFDTD":"SS"}
    # draw_miss_rate(real_class,data_path,data_path)
    miss_data=dict()
    for i in real_class.keys():
        file=data_path+"/"+i+"/miss_rate"
        miss_data[i]=read(file)

    find_best_para(miss_data,real_class)