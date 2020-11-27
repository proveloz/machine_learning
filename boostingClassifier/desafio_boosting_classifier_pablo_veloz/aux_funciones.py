import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model

from sklearn.metrics import median_absolute_error, mean_squared_error,r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
import statsmodels.formula.api as smf
import factor_analyzer as factor
import missingno as msngo
import warnings
warnings.filterwarnings('ignore')


def plot_importance(fit_model, feat_names):
    tmp_importance = fit_model.feature_importances_
    
    importance_df = pd.DataFrame(tmp_importance, index=feat_names, columns=['value'])
    importance_df = importance_df.sort_values(by='value', ascending=False).head(10)
    plt.title('Feature importance')
    importance_df['value'].plot.barh()
    return importance_df.index



def bar_counts(dataframe,lista,dim,tam):
    plt.figure(figsize=(tam[0],tam[1]))
    for n, i in enumerate(lista):
        plt.subplot(dim[0], dim[1], n + 1)
        sns.countplot(y= dataframe[i],order = dataframe[i].value_counts().index)
        plt.title('Frecuencias para {}'.format(i))
        plt.tight_layout()

def distplot_target_sep(dataframe,lista,target,dim,tam):
    var=lista[1:]
    plt.figure(figsize=(tam[0],tam[1]))
    plt.subplots_adjust(wspace=0.4,right = 2.0,bottom = -8.0)
    for i,n in enumerate(var):
        if n!=target:
            plt.subplot(dim[0],dim[1],i+1)
            sns.distplot(dataframe[n][dataframe[target]==0],kde_kws={"label": "0"})
            sns.distplot(dataframe[n][dataframe[target]==1],kde_kws={"label": "1"})
            plt.title(n,fontsize=15)
            sns.set_palette("bright")  

def distplot_num(dataframe,lista,target):
    var=lista
    fig = plt.figure()
    plt.subplots_adjust(wspace=0.4,right = 2.0,bottom = -4.5)
    for i,n in enumerate(var):
        if n!=target:
            plt.subplot(5,3,i+1)
            ax=sns.distplot(dataframe[n])
            plt.title(n,fontsize=15)
            sns.set_palette("bright")        
	    #ax.set_ylabel('')    
            #ax.set_xlabel('')



def boxplot_num(dataframe,lista):
    var=lista
    fig = plt.figure()
    plt.subplots_adjust(wspace=0.4,right = 2.0,bottom = -3.8)
    for i,n in enumerate(var):
        if n!="quality":
            plt.subplot(4,3,i+1)
            sns.boxplot(dataframe[n])
            plt.title(n,fontsize=20)
            sns.set_palette("bright")

def plot_bar(df):        
    lista=[df.groupby('income').workclass_recod,df.groupby('income').educ_recod,
           df.groupby('income').educational_num,df.groupby('income').civstatus,
           df.groupby('income').collars,df.groupby('income').relationship,
           df.groupby('income').race,df.groupby('income').gender,
           df.groupby('income').region]
    var=["workclass_recod","educ_recod","educational_num","civstatus","collars","relationship","race","gender","region"]
    fig, axarr = plt.subplots(3, 3, figsize=(15, 8))
    plt.subplots_adjust(wspace=0.4,right = 0.7,bottom = -1.4,hspace = 0.5)
    j=0
    k=0
    for i,n in enumerate(lista):
        #plot.subplot(3,3,i+1)
        lista[i].value_counts().unstack(0).plot.bar(ax=axarr[j][k])
        axarr[j][k].set_title(var[i], fontsize=18)
        j+=1
        if j==3:
            k+=1
            j=0
            if k==3:
                k=0
                j=0        
        


        
        
def hist_box(df,col):
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)})
    mean=df[col].mean()
    median=df[col].median()
    mode=df[col].mode().get_values()[0]
    
    sns.boxplot(df[col], ax=ax_box)
    
    ax_box.axvline(mean, color='r', linestyle='--')
    ax_box.axvline(median, color='g', linestyle='-')
    ax_box.axvline(mode, color='b', linestyle='-')

    sns.distplot(df[col], ax=ax_hist)
    ax_hist.axvline(mean, color='r', linestyle='--')
    ax_hist.axvline(median, color='g', linestyle='-')
    ax_hist.axvline(mode, color='b', linestyle='-')
    
    plt.legend({'Mean':mean,'Median':median,'Mode':mode})
    ax_box.set(xlabel=str(col)+": "+"Mean: "+str(round(mean,2))+" Median: "+str(round(median,2)))
    
    plt.show()

def cat_plot(varx,vary,var_grouped):
    bins=[0,9,20,30,75]
    names=["0-9","10-20","21-30","31-75"]
    df_copy["absences"]=pd.cut(df_copy["absences"],bins,labels=names)
    sns.catplot(x="sex", y="G1", hue="absences", data=df_copy,height=5, kind="bar", palette="muted")
    
    
    
def plot_columns_behaviour(df, kind='countplot'):
    """Plots the columns of the given dataframe using a countplot or a distplot
    Parameters
    ----------
    df : DataFrame
    kind : str
        countplot or distplot
    """

    cols = list(df.columns)
    n_cols = 3
    n_rows = np.ceil(len(cols) / n_cols)
    plt.figure(figsize=(n_cols * 5, 5 * n_rows))

    for n, col_name in enumerate(cols):
        plt.subplot(n_rows, n_cols, n + 1)

        col = df[col_name]

        if kind == 'countplot':
            sns.countplot(y=col)
            plt.title(humanize(col_name))
            plt.xlabel("")
        else:
            sns.distplot(col, rug=True)
            plt.title(humanize(col_name))
            plt.xlabel("")
            plt.axvline(col.mean(), color='tomato',
                        linestyle='--', label='mean')
            plt.axvline(col.median(), color='green',
                        linestyle='--', label='median')
            plt.legend()
        plt.tight_layout()    
    
    
    
def show_correlaciones(df, value=0.7):
    plt.figure(figsize=(15, 6))
    M = df.corr()
    value_corr = M[((M > value) & (M < 1) | (M < -value))
                  ].dropna(axis=0, how='all').dropna(axis=1, how='all')
    ax=sns.heatmap(value_corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 320, n=250),annot=True)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')
    plt.title("Corralaciones Totales")
    
def binarize(df):
    #df_dummy=df.copy()
    #df_dummy.columns
    #df_dummy.dtypes
    lista_cat=[]
    for i in df.columns:
        tipo_col=df[i].dtype
        if tipo_col==object:
            lista_cat.append(i)
    #print(lista_cat)
    for i in lista_cat:
        df = pd.get_dummies(df, columns=[i],drop_first=True)
    return df

def string_col_for_models(df,var):
    """probar con largo-2"""
    string_cols=var+"~"
    largo=len(df_dummy.columns)
    for i,n in enumerate(df.columns):
        if n==var:
            pass
        else:
            if i!=largo-1:
                string_cols+=n+"+"

            else:
                string_cols+=n
    return string_cols

def string_col_new_for_model(df,lista_vars):
    string_cols=var+"~"
    lista=lista_vars
    largo=len(df.columns)
    for i,n in enumerate(df.columns):
        if n in lista:
            pass
        else:
            if i!=largo-1:
                string_cols+=n+"+"

            else:
                string_cols+=n
    return string_cols


def significant_pvalues(model):
    """Returns the significant pvalues in model (95% significance)"""
    pvalues = model.pvalues[1:]
    return pvalues[pvalues <= 0.025]

def compara_test_predict(y_test,y_predict):

    df3 = pd.DataFrame({'Actual': y_test, 'Prediccion': y_predict})
    df4 = df3.head(25)
    df4.plot(kind='bar',figsize=(16,10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title("Posicion del dato vs valor en modelo test - prediccion")
    plt.show()
    
    
#def predict_target(df_dummy,var,stdScaler=True,model):
#    y_vec=df_dummy[var]
#    X_mat=df_dummy.drop(var,axis=1)
#    X_train,X_test,y_train,y_test=train_test_split(X_mat,y_vec,test_size=0.33,random_state=1612399)
#    if std_scaler==True:
#        std_scaler=StandardScaler().fit(X_train)
#        data_preproc_Xtrain=std_scaler.transform(X_train)
#        data_preproc_Xtest=std_scaler.transform(X_test)
#        if model=="LinReg":
#            model_1=data_preproc_Xtrain.LinearRegression(fit_intercept=True, normalize=False)
#            modelo1_entrenado=model_1.fit(X_train,y_train)
#            yhat=modelo1_entrenado.predict(X_test)
#    return y_hat

#def report_metrics(model, X_test, y_test):
#    print('Test R^2 accuracy: {0}'.format(r2_score(y_test,model.predict(X_test)).round(3)))
#    a=r2_score(y_test,model.predict(X_test)).round(3))
#    print('Test RMSE accuracy: {0}'.format(np.sqrt(mean_squared_error(y_test,model.predict(X_test))).round(3)))
#    b=np.sqrt(mean_squared_error(y_test,model.predict(X_test))).round(3))
#    print('Test MAE accuracy: {0}'.format(median_absolute_error(y_test,model.predict(X_test)).round(3)))
#    c=median_absolute_error(y_test,model.predict(X_test)).round(3))
#    lista=[a,b,c]
#    return lista
    
def dependencia_parcial(model):
    model.fit(X_train, y_train)
    x_grid = generate_X_grid(model)
    attribute = X_train_df.columns
    cols = 2 
    rows = np.ceil(len(attribute)/cols)
    plt.figure(figsize=(10, 12))

    for i, n in enumerate(range(len(attribute))):
        plt.subplot(rows, cols, i + 1)
        partial_dep, confidence_intervals = model.partial_dependence(x_grid,feature = i + 1, width=.95)
        plt.plot(x_grid[:, n], partial_dep, color='dodgerblue')
        plt.fill_between(x_grid[:, n],confidence_intervals[0][:, 0],confidence_intervals[0][:, 1],color='dodgerblue', alpha=.25)
        plt.title(attribute[n])
        plt.plot(X_train_df[attribute[n]],[plt.ylim()[0]] * len(X_train_df[attribute[n]]),'|', color='orange', alpha=.5)
    plt.tight_layout()


def matriz_confusion(y_test, y_hat,labels):
    from sklearn.metrics import confusion_matrix
    cnf = confusion_matrix(y_test, y_hat) / len(y_test)
    tmp=sns.heatmap(cnf, xticklabels=labels, yticklabels=labels, 
                annot=True, fmt=".1%", cbar=False, cmap='Blues');
    return tmp



def histogram_overlap(df, attribute, target, perc=100):
    # get lower bound
    empirical_lower_bound = np.floor(df[attribute].min())
    # get upper bound
    empirical_upper_bound = np.ceil(df[attribute].max())
    # preserve histograms
    tmp_hist_holder = dict()
    # for each target class
    for unique_value in np.unique(df[target]):
        # get histogram
        tmp, _ = np.histogram(
        # for a specific attribute
        df[df[target] == unique_value][attribute],
        # define percentage
        bins=perc,
        # limit empirical range for comparison
        range=[empirical_lower_bound, empirical_upper_bound]
        )
        # append to dict
        tmp_hist_holder["h_"+str(unique_value)] = tmp
    get_minima = np.minimum(tmp_hist_holder['h_1'], tmp_hist_holder['h_0'])
    get_maxima = tmp_hist_holder['h_0']
    if np.array_equal(get_minima, tmp_hist_holder['h_0']):
        get_maxima = tmp_hist_holder['h_1']
    intersection = np.true_divide(np.sum(get_minima), np.sum(get_maxima))
    return intersection



