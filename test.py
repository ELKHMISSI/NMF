#%%
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# %%
Books = pd.read_csv('Books.csv',sep=';')
Users = pd.read_csv('Users.csv',sep=";")
Ratings = pd.read_csv('BX_Ratings.csv',sep=";",encoding="latin1")
Books=Books.drop(["Image-URL-S","Image-URL-M","Image-URL-L","Publisher","Year-Of-Publication"],axis=1)
Ratings=Ratings.drop(Ratings[Ratings["ISBN"].isin(["0373761619","0735201994",
                            "0330482750","0413326608","0440500702","0373166982",
                            "0894805959","8423920143","034050823X","039482492X",
                            "0553570722","096401811X","085409878X","1874100055",
                            "0006479839","0807735132","0394720784","0723245827",
                            "1581801653","006263545X"])].index)
Books_rate = Ratings.merge(Books,how="left",on="ISBN")
Books_rate = Books_rate.dropna()
# %%

U=Books_rate.groupby("User-ID")["Book-Rating"].count()
U=U.loc[U>300].index.values
B=Books_rate.groupby("ISBN")["Book-Rating"].count()
B=B.loc[B>200].index.values
F=Books_rate.loc[Books_rate["User-ID"].isin(U)&Books_rate["ISBN"].isin(B)]
# %%
Xnan = pd.pivot_table(F,index="User-ID",columns="ISBN",values="Book-Rating")
Xnan.shape
X=Xnan.fillna(0)
X.shape
# %%
X_fact=NMF(n_components=20,beta_loss='frobenius',max_iter=1000)
W=X_fact.fit_transform(X)
H=X_fact.components_
# %%
X_hat = W@H
X_hat=pd.DataFrame(X_hat)
X_hat[X_hat>10]=10
X_hat[X_hat<1]=1
X_hat=round(X_hat)
X_hat.columns=X.columns
X_hat.index=X.index
# %%
def norm_frob(A) :
    A=np.asmatrix(A)
    return float(np.sqrt((np.transpose(A)@A).trace()))
    
def NMF_gen(X,r,alpha):
    X_fact=NMF(n_components=20,beta_loss='frobenius',alpha=alpha)
    W=X_fact.fit_transform(X)
    H=X_fact.components_
    return W@H

def best_model(X,r_min,r_max,alpha_min,alpha_max):
    vect_r=range(r_min,r_max+1)
    vect_alpha=np.arange(alpha_min,alpha_max+0.01,0.01)
    err=np.zeros((len(vect_r),len(vect_alpha)))
    for i in range(len(vect_r)):
        for j in range(len(vect_alpha)):
            X_hat=NMF_gen(X,i,j)
            err[i,j]=norm_frob(X-X_hat)
    err = pd.DataFrame(err,index=vect_r,columns=vect_alpha)
    return err
# %%
r_min,r_max,alpha_min,alpha_max=2,30,0.02,0.3
vect_r=range(r_min,r_max+1)
vect_alpha=np.arange(alpha_min,alpha_max+0.01,0.01)
r_min,r_max,alpha_min,alpha_max,vect_alpha,vect_r
# %%
M=best_model(X,r_min,r_max,alpha_min,alpha_max)
# %%
# creating 3d plot using matplotlib
# in python

# for creating a responsive plot
%matplotlib widget
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# creating random dataset
xs = vect_alpha

ys = vect_r

zs = range(29)

# creating figure
fig = plt.figure()
ax = Axes3D(fig)

# creating the plot
plot_geeks = ax.scatter(xs, ys, zs, color='green')

# setting title and labels
ax.set_title("3D plot")
ax.set_xlabel('alpha')
ax.set_ylabel('r')
ax.set_zlabel('RMSE')

# displaying the plot
plt.show()

# %%
import plotly.graph_objects as go

fig = go.Figure(go.Surface(
    x = np.asarray(vect_alpha),
    y = np.asarray(vect_r),
    z = np.asarray(M)))
fig.update_layout(
        scene = {
            "xaxis": {"nticks": 20},
            "zaxis": {"nticks": 4},
            'camera_eye': {"x": 0, "y": -1, "z": 0.5},
            "aspectratio": {"x": 1, "y": 1, "z": 0.2}
        })
fig.show()
# %%
