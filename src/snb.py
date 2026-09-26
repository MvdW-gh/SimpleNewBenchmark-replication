"""
Code for "A Simple New Benchmark for Forecasting Bond Risk Premia".

All estimation code: out-of-sample forecasts with principal-component regressions, benchmark models and
neural networks, Clark-West tests and economic value, all computed in pc_regression().

This is a local version of the authors' original implementation (a Jupyter notebook for the linear models and a
Databricks notebook for the neural networks). Comments that refer to "the original" describe the changes made
relative to that implementation: replacing Databricks/Spark by local equivalents, and a few corrections.
"""

import pandas as pd
import numpy as np 
from numpy.linalg import inv
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from numpy.linalg import eig
import statsmodels.api as sm
import statsmodels.formula.api as smf
from dateutil import relativedelta
from statsmodels.nonparametric import kernel_regression as kr
from statsmodels.nonparametric.kde import kernel_switch
from statsmodels.nonparametric.kde import KDEUnivariate as kde
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import t as tstat, kurtosis
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
pd.set_option('display.max_rows', 100)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data(data_dir=DATA_DIR):
    """Load yields, macro data and CPI from the local data folder (replaces the Spark/Azure reads)."""
    data_dir = Path(data_dir)

    yields = pd.read_csv(data_dir / "LW_monthly.csv", encoding="utf-8-sig").set_index("Year").astype(float)
    yields.index = pd.to_datetime(list(yields.index.astype(int).astype(str)),format="%Y%m")
    yields.columns=[i for i in range(1,len(yields.columns)+1)]
    yields.columns.name='Maturity'

    macro = pd.read_csv(data_dir / "data_mccracken.csv", header=None, encoding="utf-8-sig").astype(float)
    macro.index = pd.date_range(start='3/1/1959', periods=len(macro),freq='MS')
    macro.columns=list(range(len(macro.columns)))

    # get cieslak and povala data
    cpi = pd.read_csv(data_dir / "CPI data for cieslak and povala (FRED download).csv", encoding="utf-8-sig")
    cpi = cpi.rename(columns={"DATE":"date"}).set_index("date").astype(float)['CPILFESL']
    cpi.index = pd.to_datetime(list(cpi.index),format="%Y-%m-%d")

    return yields, macro, cpi

# COMMAND ----------

def simulate_yields(yields, other_predictors, n_components = 3, num_simulations =100):
    X=yields.copy()
    other_predictors=other_predictors.copy()
    # remove nans
    idx = ~np.isnan(X).any(axis=1) & ~np.isnan(other_predictors).any(axis=1)    
    X = X[idx]
    other_predictors=other_predictors[idx]
    
    values,vectors=eig(np.cov(X,rowvar=False))
    
    # make sure the sign is consistent with level, slope and curvature
    if vectors[0,0]<0:
        vectors[:,0]=vectors[:,0]*-1
    if vectors[0,1]>0:
        vectors[:,1]=vectors[:,1]*-1
    if vectors[0,2]>0:
        vectors[:,2]=vectors[:,2]*-1
    
    # get principal components
    X_pc=X@vectors[:,:n_components]

    # linear regression
    y_1 = X_pc[1:,:]
    n = len(y_1)
    X_1 = X_pc[:-1]
    model = LinearRegression()
    model.fit(X_1,y_1)
    # Get predictions
    y_pred = model.predict(X_1)
    
    # Calculate residuals
    residuals = y_1 - y_pred
    
    # Calculate residual standard error (standard deviation of error terms)
    k = X_1.shape[1] + 1  # number of features + intercept
    df_resid = n - k    # degrees of freedom
    
    # Standard deviation of error terms
    se = np.sqrt(np.sum(residuals**2,axis=0) / df_resid)

    # randomly draw error terms
    random_row_indices = np.random.randint(0, len(residuals), size=(len(X_pc),num_simulations))
    error_terms = residuals[random_row_indices].transpose(0, 2, 1)

    simulated_pcs= np.random.normal(size = (len(X_pc), len(se), num_simulations))*se.reshape(1,-1,1)+model.intercept_.reshape(1,-1,1)
    simulated_pcs= error_terms+model.intercept_.reshape(1,-1,1)
     
    simulated_pcs[0,:,:] = X_pc[0,:].reshape(-1,1)
    for simulation_t in range(1,len(X_pc)):
        simulated_pcs[simulation_t,:,:] = simulated_pcs[simulation_t,:,:] + model.coef_@simulated_pcs[simulation_t-1,:,:]

    # other predictor var model
    y_1_alternative = other_predictors[1:,:]
    X_1_alternative = other_predictors[:-1,:]
    model = LinearRegression()
    model.fit(X_1_alternative,y_1_alternative)
    y_pred = model.predict(X_1_alternative)
    residuals = y_1_alternative - y_pred
    error_terms = residuals[random_row_indices].transpose(0, 2, 1)
    simulated_other_predictors = error_terms+model.intercept_.reshape(1,-1,1)
    
    
    simulated_other_predictors[0,:,:] = other_predictors[0,:].reshape(-1,1)
    for simulation_t in range(1,len(X_pc)):
        simulated_other_predictors[simulation_t,:,:] = (simulated_other_predictors[simulation_t,:,:] + 
        model.coef_@simulated_other_predictors[simulation_t-1,:,:])

    # regression of yields on pcs
    # model = LinearRegression(fit_intercept=False)
    X_2 = X_pc
    y_2 = X
    # n = len(y_2)
    # model.fit(X_2, y_2)
    # y_pred = model.predict(X_2)
    y_pred = X_pc@vectors[:,:n_components].T
    residuals = y_2 - y_pred
    k = X_2.shape[1] + 1  # number of features + intercept
    df_resid = n - k    # degrees of freedom
    #se = np.sqrt(np.sum(residuals**2,axis=0) / df_resid)
    se = np.std(residuals, axis=0, ddof = n-k)

    # print(np.einsum('ijk,jl->ilk', simulated_pcs, vectors[:,:n_components].T)[:,:,3])
    # print(se)

    # simulated_yields = (np.random.normal(size = (len(X_pc), len(se), num_simulations))*se.reshape(1,-1,1) 
    # + simulated_pcs@((vectors[:,:n_components].T)[:,:,np.newaxis] ))
    simulated_yields = (np.random.normal(size = (len(X_pc), len(se), num_simulations))*se.reshape(1,-1,1) 
    + np.einsum('ijk,jl->ilk', simulated_pcs, vectors[:,:n_components].T))

    # add back the nans
    simulated_yields_to_return = np.empty(shape=(len(yields),simulated_yields.shape[1],simulated_yields.shape[2]))
    simulated_yields_to_return[:,:,:] = np.nan
    simulated_yields_to_return[idx,:,:] = simulated_yields[:,:,:]
    simulated_other_predictors_to_return = np.empty(shape=(len(yields),simulated_other_predictors.shape[1],simulated_other_predictors.shape[2]))
    simulated_other_predictors_to_return[:,:,:] = np.nan
    simulated_other_predictors_to_return[idx,:,:] = simulated_other_predictors[:,:,:]

    return simulated_yields_to_return, simulated_other_predictors_to_return



# COMMAND ----------

def simulate_excess_returns(excess_returns, other_predictors, n_components = 3, num_simulations =100):
    X=excess_returns.copy().astype(float)
    other_predictors=other_predictors.copy().astype(float)
    # remove nans
    idx = ~np.isnan(X).any(axis=1) & ~np.isnan(other_predictors).any(axis=1)    
    X = X[idx]
    other_predictors=other_predictors[idx]
 

    # linear regression
    y_pred = X[1:].copy()
    y_pred[:,:] = np.nan
    for col_idx in range(X.shape[1]):
        y_1 = X[1:,col_idx:col_idx+1]
        n = len(y_1)
        X_1 = X[:-1, col_idx:col_idx+1]
        model = LinearRegression()
        model.fit(X_1,y_1)
        # Get predictions
        y_pred[:,col_idx:col_idx+1] = model.predict(X_1)
    
    # Calculate residuals
    residuals = X[1:] - y_pred

    # randomly draw error terms
    random_row_indices = np.random.randint(0, len(residuals), size=(len(X),num_simulations))
    error_terms = residuals[random_row_indices].transpose(0, 2, 1)
    simulated_returns= error_terms+model.intercept_.reshape(1,-1,1)
     
    simulated_returns[0,:,:] = X[0,:].reshape(-1,1)
    for simulation_t in range(1,len(X)):
        simulated_returns[simulation_t,:,:] = simulated_returns[simulation_t,:,:] + model.coef_@simulated_returns[simulation_t-1,:,:]

    # other predictor var model
    y_1_alternative = other_predictors[1:,:]
    X_1_alternative = other_predictors[:-1,:]
    model = LinearRegression()
    model.fit(X_1_alternative,y_1_alternative)
    y_pred = model.predict(X_1_alternative)
    residuals = y_1_alternative - y_pred
    error_terms = residuals[random_row_indices].transpose(0, 2, 1)
    simulated_other_predictors = error_terms+model.intercept_.reshape(1,-1,1)
    
    
    simulated_other_predictors[0,:,:] = other_predictors[0,:].reshape(-1,1)
    for simulation_t in range(1,len(X_pc)):
        simulated_other_predictors[simulation_t,:,:] = (simulated_other_predictors[simulation_t,:,:] + 
        model.coef_@simulated_other_predictors[simulation_t-1,:,:])

    # add back the nans
    simulated_returns_to_return = np.empty(shape=(len(excess_returns),simulated_returns.shape[1],simulated_returns.shape[2]))
    simulated_returns_to_return[:,:,:] = np.nan
    simulated_returns_to_return[idx,:,:] = simulated_returns[:,:,:]
    simulated_other_predictors_to_return = np.empty(shape=(len(excess_returns),simulated_other_predictors.shape[1],simulated_other_predictors.shape[2]))
    simulated_other_predictors_to_return[:,:,:] = np.nan
    simulated_other_predictors_to_return[idx,:,:] = simulated_other_predictors[:,:,:]

    return simulated_returns_to_return, simulated_other_predictors_to_return


# COMMAND ----------

def write_to_excel(outp, excel_path):
    """Write all table-like outputs to one Excel file (local replacement of the dbfs-based original)."""
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    used_names = set()
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        for key in list(outp.keys()):
            df=outp[key]
            if not isinstance(df, (pd.DataFrame, pd.Series)):
                continue
            sheet_name=key
            sheet_name=sheet_name.replace("ields","")
            sheet_name=sheet_name.replace("orwards","")
            sheet_name=sheet_name.replace("ield","")
            sheet_name=sheet_name.replace("orward","")
            sheet_name=sheet_name.replace("changes","ch")
            sheet_name=sheet_name.replace("macro","m")
            sheet_name=sheet_name.replace("full sample regression","reg")
            sheet_name=sheet_name.replace("revised","rev")
            sheet_name=sheet_name.replace("vintage","vin")
            sheet_name=sheet_name.replace(":"," -")
            sheet_name=sheet_name.replace("+ ","")
            sheet_name=sheet_name.replace("best hyperparams", "hyp")
            sheet_name=sheet_name.replace("best val losses", "loss")
            sheet_name=sheet_name.replace("val losses", "loss")
            sheet_name=sheet_name.replace("Trading returns", "tr")
            sheet_name = sheet_name.replace("Utilities", "u")
            sheet_name=sheet_name.replace("Cieslak Povala", "CiPo")
            sheet_name=sheet_name.replace("Cochrane Piazessi", "CoPi")
            sheet_name=sheet_name.replace("Diebold Li", "DiLi")
            sheet_name=sheet_name.replace("param ", "")	
            sheet_name=sheet_name.replace("PC correlations", "pccor")
            if "NN" in sheet_name:
                sheet_name.replace(" (rev)","")

            # Excel limits sheet names to 31 characters and requires them to be unique
            sheet_name = sheet_name[:31]
            suffix = 1
            while sheet_name.lower() in used_names:
                sheet_name = f"{sheet_name[:28]}~{suffix}"
                suffix += 1
            used_names.add(sheet_name.lower())

            df.to_excel(writer,sheet_name=sheet_name)


def save_figures(outp, folder):
    """Save all matplotlib figures in the output dictionary as PNG files."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for key, val in outp.items():
        if isinstance(val, plt.Figure):
            filename = "".join(c if c.isalnum() or c in " -_()," else "_" for c in key).strip()
            try:
                val.savefig(folder / f"{filename}.png", bbox_inches="tight")
            except OSError as e:
                # e.g. Windows' 260-character path limit when the package sits in a deep folder
                print(f"warning: figure '{key}' not saved ({e.__class__.__name__}); results are not affected")



def cieslak_povala_single_factor(X,df_inf,y,holding_period = 12,maturities=[24,36,48,60,84,120]):
    # remove nans
    idx = ~np.isnan(X).any(axis=1)
    X = X[idx]
    y = y[idx]
    df_inf = df_inf[idx]
    
    # mat-specific cycle    
    ct = X-LinearRegression().fit(df_inf,X).predict(df_inf)
    ct_bar = np.expand_dims(np.mean(X[:,1:],axis=1),axis=-1)
    
    rx_bar = np.zeros((len(y),1))
    for col in range(len(maturities)):
        annualized_mat = maturities[col]/12
        rx_bar = rx_bar+y[:,col:col+1]/annualized_mat
    rx_bar = rx_bar/(len(maturities))
    
    indep_var = np.concatenate((ct_bar,ct[:,:1]),axis=1)
    reg = LinearRegression().fit(indep_var[:-holding_period],rx_bar[holding_period:])
    fitted_rx_bar = reg.predict(indep_var)
    
    reg = LinearRegression().fit(fitted_rx_bar[holding_period:],y[holding_period:])
    
    if len(X)%50==0:
        
        fig = plt.figure(dpi = 200)
        plt.plot(ct_bar)
        plt.legend(labels=['ct_bar'])
        plt.title("ct_bar")
        plt.close(fig)
               
        fig2 = plt.figure(dpi = 200)
        plt.plot(fitted_rx_bar)
        plt.legend(labels=['cf_t'])
        plt.title("cf_t")
        plt.close(fig2)
    else:
        fig = None
        fig2 = None

    return reg.predict(fitted_rx_bar[-1:]), fig, fig2
    
    

def cochrane_piazessi(X,y,holding_period=12, return_cp_factor = False,return_cp_factor_only = False):
    # remove nans
    idx = ~np.isnan(X).any(axis=1)
    X = X[idx]
    y = y[idx]
    reg=LinearRegression().fit(X[:-holding_period],y[holding_period:,-1:])
    X_fit = reg.predict(X[:-holding_period])
    if return_cp_factor:
        return X_fit
    X_fit_oos = reg.predict(X[-1:,:])
    n_maturities = y.shape[1]
    pred = np.zeros((1,n_maturities))
    for mat in range(n_maturities):
        reg2 = LinearRegression()
        pred[0,mat] = reg2.fit(X_fit,y[holding_period:,mat]).predict(X_fit_oos)
    if return_cp_factor_only:
        return pred, X_fit
    return pred
    
def direct_regression(X,y,holding_period=12):
    # remove nans
    idx = ~np.isnan(X).any(axis=1)
    X = X[idx]
    y = y[idx]
    
    reg=LinearRegression().fit(X[:-holding_period],y[holding_period:]) #sklearn regression has default to add intercept
    return reg.predict(X[-1:,:])
    
def direct_regression_coef(X,y,holding_period=12):
    idx = ~np.isnan(X).any(axis=1)
    X = X[idx]
    y = y[idx]
    
    reg=sm.OLS(y[holding_period:],np.concatenate((np.ones((len(X)-holding_period,1)),X[:-holding_period]),axis=1))
    if holding_period>0:
        res=reg.fit(cov_type='HAC',cov_kwds={'maxlags':holding_period-1})
    else:
        res = reg.fit(cov_type='HAC')
    adj_r2 = res.rsquared_adj
    var=res.cov_params()
    pars=res.params
    se = np.empty(pars.shape)
    for i in range(len(se)):
        se[i]=np.sqrt(var[i,i])
    
    t_score=pars/se
    pvals = np.empty(t_score.shape)
    for i in range(len(pvals)):
        pvals[i] = 2*tstat.cdf(-abs(t_score[i]),df=len(X)-1)
        
    # due to a bug in statsmodels, HAC cov not fully supported. need to rerun with normal se to get R2. 
    reg=LinearRegression().fit(X[:-holding_period],y[holding_period:])
    r2=reg.score(X[:-holding_period,],y[holding_period:])
    
    return pars, se, t_score, pvals, r2, adj_r2


def cieslak_povala_pc(X_comb,y,n_components=3,holding_period=12,skip_first_components = 0):
    # remove nans
    idx = ~np.isnan(X_comb).any(axis=1)
    X_comb = X_comb[idx]
    X = X_comb[:,:-1]
    inf_arr = X_comb[:,-1:]
    y = y[idx]
    
    # standardize data before taking PCs (makes a small difference)
    st=StandardScaler(with_std=False)
    
    # Compute the principal components of in sample
    pca=PCA(n_components=n_components)
    X_pc=pca.fit_transform(st.fit_transform(X))
    
    # Compute principal components to be used for prediction. Note that if hp>1, we must skip periods between 
    # training and predicting due to overlapping returns. However, the explanatory variables during these 
    # skip periods can still be used to fit the principal components. 
    X_to_fit = np.concatenate((X_pc[:,skip_first_components:n_components],inf_arr),axis=1)
    reg=LinearRegression().fit(X_to_fit[:-holding_period],y[holding_period:])
    return reg.predict(X_to_fit[-1:])

def pc(X,y,n_components=3,holding_period=12,skip_first_components = 0):
    """
    Returns the prediction based on a regression on the PCs. 

    Parameters
    ----------
    X : Forward rates.
    y : Excess returns.
    n_components : Number of principal components.
    holding_period: holding period

    Returns
    A prediction for the excess return.

    """
    # remove nans
    idx = ~np.isnan(X).any(axis=1)
    X = X[idx]
    y = y[idx]
    
    # standardize data before taking PCs (makes a small difference)
    st=StandardScaler(with_std=False)
    
    # Compute the principal components of in sample
    pca=PCA(n_components=n_components)
    X_pc=pca.fit_transform(st.fit_transform(X))
    
    # Compute principal components to be used for prediction. Note that if hp>1, we must skip periods between 
    # training and predicting due to overlapping returns. However, the explanatory variables during these 
    # skip periods can still be used to fit the principal components. 
    reg=LinearRegression().fit(X_pc[:-holding_period,skip_first_components:n_components],y[holding_period:])
    return reg.predict(X_pc[-1:,skip_first_components:n_components])

def pc_double(X, X2 ,y,n_components=3,holding_period=12,skip_first_components = 0):
    """
    Returns the prediction based on a regression on two sets of PCs. 

    Parameters
    ----------
    X : Forward rates.
    y : Excess returns.
    n_components : Number of principal components.
    holding_period: holding period

    Returns
    A prediction for the excess return.

    """
    # remove nans
    idx = ~np.isnan(np.concatenate((X,X2),axis=1)).any(axis=1)
    X = X[idx].copy()
    X2 = X2[idx].copy()
    y = y[idx].copy()
    
    # standardize data before taking PCs (makes a small difference)
    st=StandardScaler(with_std=False)
    
    # Compute the principal components of in sample
    pca=PCA(n_components=n_components)
    X_pc=pca.fit_transform(st.fit_transform(X))
    X_pc2=pca.fit_transform(st.fit_transform(X2))
    
    # Compute principal components to be used for prediction. Note that if hp>1, we must skip periods between 
    # training and predicting due to overlapping returns. However, the explanatory variables during these 
    # skip periods can still be used to fit the principal components. 
    reg=LinearRegression().fit(np.concatenate((X_pc[:-holding_period,skip_first_components:n_components], 
                                               X_pc2[:-holding_period,skip_first_components:n_components]),axis=1),y[holding_period:])
    return reg.predict(np.concatenate((X_pc[-1:,skip_first_components:n_components],X_pc2[-1:,skip_first_components:n_components]),axis=1))

def pc_coef(X,y,n_components=3,holding_period=12, take_pc = True, robust_se = True):
    """
    Returns the coefficients and standard errors of PC regressions. 

    Parameters
    ----------
    X : Forward rates.
    X_oos : Forward rates to use for prediction.
    y : Excess returns.
    n_components : Number of principal components.

    Reurns
    A prediction for the excess return.

    """
    # remove nans
    X_incl_nan = X.copy()
    idx = ~np.isnan(X).any(axis=1)
    X = X[idx]
    y = y[idx]
    
    # standardize data before taking PCs (makes a small difference)
    if take_pc:
        st=StandardScaler(with_std=False)
        
        # Compute the principal components
        X = st.fit_transform(X)
    
    values,vectors=eig(np.cov(X,rowvar=False))
    
    # make sure the sign is consistent with level, slope and curvature
    if vectors[0,0]<0:
        vectors[:,0]=vectors[:,0]*-1
    if vectors[0,1]>0:
        vectors[:,1]=vectors[:,1]*-1
    if vectors[0,2]>0:
        vectors[:,2]=vectors[:,2]*-1

    if take_pc:
        
        fitted=X@vectors[:,:n_components]
        fitted_incl_nan = X_incl_nan@vectors[:,:n_components]
    else:
        fitted=X
        fitted_incl_nan = X_incl_nan
    
    # reg=sm.OLS(y[holding_period:],np.concatenate((np.ones((len(X)-holding_period,1)),fitted[:-holding_period,:n_components]),axis=1))
    reg=sm.OLS(y[holding_period:],np.concatenate((np.ones((len(X)-holding_period,1)),fitted[:-holding_period,:]),axis=1))
    if holding_period>0:
        res=reg.fit(cov_type='HAC',cov_kwds={'maxlags':holding_period-1})
    else:
        res = reg.fit(cov_type='HAC')
    if not robust_se:
        res=reg.fit()
    var=res.cov_params()
    pars=res.params
    se = np.empty(pars.shape)
    for i in range(len(se)):
        se[i]=np.sqrt(var[i,i])
    
    t_score=pars/se
    pvals = np.empty(t_score.shape)
    for i in range(len(pvals)):
        pvals[i] = 2*tstat.cdf(-abs(t_score[i]),df=len(X)-1)
        
    # due to a bug in statsmodels, HAC cov not fully supported. need to rerun with normal se to get R2. 
    # reg=LinearRegression(fit_intercept=False).fit(np.concatenate((np.ones((len(X)-holding_period,1)),fitted[:-holding_period,:n_components]),axis=1),y[holding_period:])
    reg=LinearRegression(fit_intercept=False).fit(np.concatenate((np.ones((len(X)-holding_period,1)),fitted[:-holding_period,:]),axis=1),y[holding_period:])
    # r2=reg.score(np.concatenate((np.ones((len(X)-holding_period,1)),fitted[:-holding_period,:n_components]),axis=1),y[holding_period:])
    r2=reg.score(np.concatenate((np.ones((len(X)-holding_period,1)),fitted[:-holding_period,:]),axis=1),y[holding_period:])
    
    return pars, se, t_score, pvals, r2, fitted_incl_nan, vectors

def pc_coef_macro(X,m,y,n_components=3,holding_period=12,m_components=8):
    """
    Returns the coefficients and standard errors of PC regressions. 

    Parameters
    ----------
    X : Forward rates.
    X_oos : Forward rates to use for prediction.
    y : Excess returns.
    n_components : Number of principal components.

    Reurns
    A prediction for the excess return.

    """
    # remove nans
    len_X = len(X)
    nonnan_idx = ~np.isnan(X[:-holding_period]).any(axis=1)
    idx = ~np.isnan(X).any(axis=1)
    X = X[idx]
    y = y[idx]
    m = m[idx]
    
    # standardize data before taking PCs (makes a small difference)
    st=StandardScaler(with_std=False)
    X = st.fit_transform(X)
    st2=StandardScaler()
    m=st2.fit_transform(m)
    
    # Compute the eigenvectors
    values,vectors=eig(np.cov(X,rowvar=False))
    
    # make sure the sign is consistent with level, slope and curvature
    if vectors[0,0]<0:
        vectors[:,0]=vectors[:,0]*-1
    if vectors[0,1]>0:
        vectors[:,1]=vectors[:,1]*-1
    if vectors[0,2]>0:
        vectors[:,2]=vectors[:,2]*-1
    
    # get principal components
    fitted=X@vectors[:,:n_components]
    
    m_values,m_vectors=eig(np.cov(m,rowvar=False))
    for i in range(m_components):
        if m_vectors[0,i]<0:
            m_vectors[:,i]=m_vectors[:,i]*-1
    m_fitted=m@m_vectors[:,:m_components]
    
    # list of principal components that are significant at 1% level 
    sig_pcs=[]
    
    # determine if pcs add significantly to prediction
    for i in range(m_components):
        reg=sm.OLS(y[holding_period:],sm.add_constant(m_fitted[:-holding_period,i:i+1]))
        if holding_period>0:
            res=reg.fit(cov_type='HAC',cov_kwds={'maxlags':holding_period-1})
        else:
            res=reg.fit(cov_type='HAC')
        var=res.cov_params()[-1,-1]
        se=np.sqrt(var)
        coef=res.params[-1]
        t_score=coef/se
        if np.abs(t_score)>tstat.ppf(0.995,m.shape[0]-n_components):
            sig_pcs.append(i)
    
    # number of significant macro components
    n_sig=len(sig_pcs)
    
    # compute BIC of all combinations of pcs. There are 2 to the power of n_sig combinations possible 
    # (each pc can be included or not). Translate i to binary form to indicate which pcs are included. 
    # e.g. suppose there are 3 pcs, then n_sig=8. Then i=6 becomes 101, so pc 1 and 3 are included and 2 not. 
    combs=np.power(2,n_sig)
    best_bic=10000 # arbitrary high number
    for i in range(combs):
        
        sig_pcs_used=[]
        
        # q keeps track of how much of i "is left"
        q=i
        
        # also include yield/forward pcs
        x=np.copy(fitted[:-holding_period])
        x_oos=fitted[-1:]
        for j in range(1,n_sig+1):
            if q>=np.power(2,n_sig-j):
                x=np.concatenate((x,m_fitted[:-holding_period,sig_pcs[-j]:sig_pcs[-j]+1]),axis=1)
                x_oos=np.concatenate((x_oos,m_fitted[-1:,sig_pcs[-j]:sig_pcs[-j]+1]),axis=1)
                q=q-np.power(2,n_sig-j)
        
                sig_pcs_used.append(sig_pcs[-j])
        
        reg=sm.OLS(y[holding_period:],x)
        if holding_period>0:
            res=reg.fit(cov_type='HAC',cov_kwds={'maxlags':11})
        else:
            es=reg.fit(cov_type='HAC')
        bic=res.bic
        
        # if bic is lower, make this the best option so far
        if bic<best_bic:
            best_bic=bic
            best_i=i
            best_x = x
            best_sig_pcs_used = sig_pcs_used
    
    reg=sm.OLS(y[holding_period:],np.concatenate((np.ones((len(X)-holding_period,1)),best_x),axis=1))
    if holding_period>0:
        res=reg.fit(cov_type='HAC',cov_kwds={'maxlags':holding_period-1})
    else:
        res=reg.fit(cov_type='HAC')
    var=res.cov_params()
    pars=res.params
    se = np.empty(pars.shape)
    for i in range(len(se)):
        se[i]=np.sqrt(var[i,i])
    
    t_score=pars/se
    pvals = np.empty(t_score.shape)
    for i in range(len(pvals)):
        pvals[i] = 2*tstat.cdf(-abs(t_score[i]),df=len(X)-1)
        
    # due to a bug in statsmodels, HAC cov not fully supported. need to rerun with normal se to get R2. 
    reg=LinearRegression(fit_intercept=False).fit(np.concatenate((np.ones((len(X)-holding_period,1)),best_x),axis=1),y[holding_period:])
    r2=reg.score(np.concatenate((np.ones((len(X)-holding_period,1)),best_x),axis=1),y[holding_period:])

    # get array inclusing missing values
    best_x_with_nan = np.empty((len_X-holding_period,best_x.shape[1]))
    best_x_with_nan[:,:]=np.nan
    best_x_with_nan[nonnan_idx,:]=best_x
    nan_arr = np.empty((holding_period,best_x.shape[1]))
    nan_arr[:,:]=np.nan
    best_x_with_nan = np.concatenate((best_x_with_nan,nan_arr),axis=0)
    
    return pars, se, t_score, pvals, r2, best_sig_pcs_used, best_x_with_nan, m_fitted, fitted, m_vectors[:,:8]
    
def pc_macro(X,X_oos,m,m_oos,y,n_components=3,m_components=8,holding_period=12):
    """
    Returns the prediction based on a regression on the PCs. 

    Parameters
    ----------
    X : Forward rates.
    m : Macro data.
    y : Excess returns.
    n_components : Number of principal components for yield data.
    m_components: Number of principal components for macro data.

    Returns
    A prediction for the excess return.

    """
    # remove nans
    idx = ~np.isnan(X).any(axis=1)
    X = X[idx]
    y = y[idx]
    m = m[idx]
    
    # standardize data before taking PCs 
    st=StandardScaler(with_std=False)
    st2=StandardScaler(with_std=True)
    
    #Compute the principal components of in sample
    pca=PCA(n_components=n_components)
    pca2=PCA(n_components=m_components)
    X_pc=pca.fit_transform(st.fit_transform(X))
    m_pc=pca2.fit_transform(st2.fit_transform(m))
    
    #Compute principal components to be used for prediction
    X_pc=sm.add_constant(X_pc)
    
    # list of principal components that are significant at 1% level 
    sig_pcs=[]
    
    # determine if pcs add significantly to prediction
    for i in range(m_components):
        reg=sm.OLS(y[holding_period:],sm.add_constant(m_pc[:-holding_period,i:i+1]))
        if holding_period>0:
            res=reg.fit(cov_type='HAC',cov_kwds={'maxlags':11})
        else:
            res=reg.fit(cov_type='HAC')
        var=res.cov_params()[-1,-1]
        se=np.sqrt(var)
        coef=res.params[-1]
        t_score=coef/se
        if np.abs(t_score)>tstat.ppf(0.995,m.shape[0]-n_components):
            sig_pcs.append(i)
    
    # number of significant macro components
    n_sig=len(sig_pcs)
    
    # compute BIC of all combinations of pcs. There are 2 to the power of n_sig combinations possible 
    # (each pc can be included or not). Translate i to binary form to indicate which pcs are include. 
    # e.g. suppose there are 3 pcs, then n_sig=8. Then i=6 becomes 101, so pc 1 and 3 are included and 2 not. 
    combs=np.power(2,n_sig)
    best_bic=10000 # arbitrary high number
    for i in range(combs):
        
        # q keeps track of how much of i "is left"
        q=i
        
        # also include yield/forward pcs
        x=np.copy(X_pc[:-holding_period])
        x_oos=X_pc[-1:]
        for j in range(1,n_sig+1):
            if q>=np.power(2,n_sig-j):
                x=np.concatenate((x,m_pc[:-holding_period,sig_pcs[-j]:sig_pcs[-j]+1]),axis=1)
                x_oos=np.concatenate((x_oos,m_pc[-1:,sig_pcs[-j]:sig_pcs[-j]+1]),axis=1)
                q=q-np.power(2,n_sig-j)
        
        reg=sm.OLS(y[holding_period:],x)
        
        if holding_period>0:
            res=reg.fit(cov_type='HAC',cov_kwds={'maxlags':holding_period-1})
        else:
            res=reg.fit(cov_type='HAC',cov_kwds={'maxlags':0})
            
        bic=res.bic
        
        # if bic is lower, make this the best option so far
        if bic<best_bic:
            best_bic=bic
            best_i=i
            y_pred=res.predict(x_oos)
    
    return y_pred[0]

def R2_pval(y_true, y_bench, y_forecast, holding_period=12):
    """
    Returns p-values of the R2_oos using Clark & West (2007) test statistic
    Adopted from code of Bianchi et al. (2020)
    """
    y_condmean=y_bench
    
    # Compute f-measure
    f = np.square(y_true-y_condmean)-np.square(y_true-y_forecast) + np.square(y_condmean-y_forecast)

    # Regress f on a constant
    x = np.ones(np.shape(f))
    model = sm.OLS(f,x, missing='drop', hasconst=True)
    
    if holding_period>0:
        res=model.fit(cov_type='HAC',cov_kwds={'maxlags':holding_period-1})
    else:
        res=model.fit(cov_type='HAC')
    var=res.cov_params()
    pars=res.params
    se = np.empty(pars.shape)
    for i in range(len(se)):
        se[i]=np.sqrt(np.float64(var[i,i]))
    
    t_score=pars/se
    
    return 1-tstat.cdf(np.float64(t_score[0]),len(y_bench)-1)

def mv(y_,y_in_,y_oos_,r_,gamma=5,alpha=0.05):
    y=y_.astype(float)/100
    y_in=y_in_.astype(float)/100
    y_oos=y_oos_.astype(float)/100
    r=r_.astype(float)/100
    w=np.zeros(y.shape)
    u=np.zeros((y.shape[0],1))
    e=np.zeros((y_in.shape[0]+y_oos.shape[0],y.shape[1]))
    for i in range(y_in.shape[0]):
        e[i,:]=y_in[i,:]-np.mean(y_in,axis=0)
    for i in range(y.shape[0]):
        e[T_in+i,:]=y[i,:]-y_oos[i,:]
        sigma=np.zeros((y.shape[1],y.shape[1]))
        for j in range(y_in.shape[0]+i-12):
            #sigma+=alpha*np.exp(-alpha*j)*np.multiply(np.ones((y.shape[1],y.shape[1])),np.transpose(e[i+y_in.shape[0]-j,:])@e[i+y_in.shape[0]-j,:])
            sigma+=alpha*np.exp(-alpha*j)*(np.transpose(e[i+y_in.shape[0]-j-12:i+y_in.shape[0]-j-11,:])@e[i+y_in.shape[0]-j-12:i+y_in.shape[0]-j-11,:])
            #print(sigma)
        #sigma=np.transpose(e[i+y_in.shape[0]-1:i+y_in.shape[0]])@e[i+y_in.shape[0]-1:i+y_in.shape[0]]

        cov_input = np.concatenate((y_in,y_oos[:i-11]),axis=0)

        idx = ~np.isnan(cov_input).any(axis=1)
        
        
        sigma=np.cov(cov_input[idx],rowvar=False)
        #w[i:i+1,:]=np.transpose(1/gamma*inv(sigma)@np.transpose(y[i:i+1,:]))
        w[i,:]=1/gamma*y[i,:]/sigma
        
        #for j in range(w.shape[1]):
        #    
        ##    if w[i,j]<-1:
         #       w[i,j]=-1
         #   elif w[i,j]>2:
         #       w[i,j]=2 
        #u[i,0]=w[i,:]@np.transpose(y_oos[i,:])-gamma/2*w[i,:]@sigma@np.transpose(w[i,:])
        #u[i,0]=1+r[i]+w[i,:]@np.transpose(y_oos[i,:])-gamma/2*np.square(w[i,:]@np.transpose(y_oos[i,:]))
        u[i,0]=1+r[i]+w[i,:]@np.transpose(y_oos[i,:])-gamma/2/(1+gamma)*np.square(1+r[i]+w[i,:]@np.transpose(y_oos[i,:]))
        #u[i,0]=1+r[i]+w[i,:]@np.transpose(y_oos[i,:])
    #print(np.mean(w))
    return u


def mv_final(predictions_df, returns_df, forwards_df,maturities,holding_period, gamma=5):
    weights = predictions_df.copy()
    sigma = predictions_df.copy()
    sigma.loc[:,:] = np.nan

    returns_df = returns_df/100
    predictions_df= predictions_df/100
    forwards_df = forwards_df/100
    

    hist_vol=returns_df.shift(periods=holding_period).expanding().var()

    weights = predictions_df/hist_vol/gamma
    weights = weights.clip(-1,2)

    returns = 1+weights*returns_df+forwards_df.shift(periods=holding_period)[[holding_period]].to_numpy()
    returns = returns.dropna()

    utilities=1+returns-0.5*gamma*((returns-returns.mean())**2)
    utilities=returns-0.5*gamma/(1+gamma)*(returns**2)

    excess_returns = (weights*returns_df).dropna()
    return utilities, weights, excess_returns

    
    
    
def cer(u,aversion=5):
    return 1/aversion+1-np.sqrt((1+1/aversion)**2-(2/aversion+2)*np.mean(u.astype(float)))-1



    

# COMMAND ----------

def pc(X,y,n_components=3,holding_period=12,skip_first_components = 0, **kwargs):
    """
    Returns the prediction based on a regression on the PCs. 

    Parameters
    ----------
    X : Forward rates.
    y : Excess returns.
    n_components : Number of principal components.
    holding_period: holding period

    Returns
    A prediction for the excess return.

    """
    # remove nans
    idx = ~np.isnan(X).any(axis=1)
    X = X[idx]
    y = y[idx]
    
    # standardize data before taking PCs (makes a small difference)
    st=StandardScaler(with_std=False)
    
    # Compute the principal components of in sample
    pca=PCA(n_components=n_components)
    X_pc=pca.fit_transform(st.fit_transform(X))
    
    # Compute principal components to be used for prediction. Note that if hp>1, we must skip periods between 
    # training and predicting due to overlapping returns. However, the explanatory variables during these 
    # skip periods can still be used to fit the principal components. 
    reg=LinearRegression().fit(X_pc[:-holding_period,skip_first_components:n_components],y[holding_period:])
    return reg.predict(X_pc[-1:,skip_first_components:n_components])

def parallel_function(parallel_input, parallel_macro, parallel_excess_ret, holding_period, t, T_in, window, function, **kwargs):
    # inputs are passed directly (the original received Spark broadcast variables and used .value)
    if window=="rolling":
        t_begin = t-T_in
    elif window =="expanding":
        t_begin = 0
    if parallel_macro is not None:
        parallel_macro = parallel_macro.iloc[t_begin:t-holding_period+1].values
    return function(parallel_input.iloc[t_begin:t-holding_period+1].values,parallel_macro,parallel_excess_ret.iloc[t_begin:t-holding_period+1].values,holding_period=holding_period, **kwargs)


# COMMAND ----------


from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization,Dropout, Input, Concatenate
from keras import optimizers,regularizers
from keras.callbacks import EarlyStopping


import os
import random

from numpy.random import seed
import tensorflow as tf

def set_seed(random_state) -> int:
    """
    Set seed.

    Returns:
        int: the seed
    """
    # Set seed for all frameworks
    # (https://odsc.medium.com/properly-setting-the-random-seed-in-ml-experiments-not-as-
    # simple-as-you-might-imagine-219969c84752)
    seed(random_state)
    tf.random.set_seed(random_state)
    random.seed(random_state)
    os.environ["PYTHONHASHSEED"] = str(random_state)

    return random_state

def ann(X,y,n,l,L,u,seed_,holding_period=12, cochrane_piazessi=False, diebold_li = False, diebold_li_direct = False, pc_nn= False):
    """
    Returns the forecast based on the ann approach

    Parameters
    ----------
    X : In sample predictors
    X_oos : Out of sample predictors
    y : In sample dependent variable
    n : number of nodes
    l : number of layers
    Returns
    -------
    Prediction of y.

    """
    
    
    # remove nans
    X= X.copy()
    y=y.copy()
    idx = ~np.isnan(X).any(axis=1)
    X = X[idx].astype(float)
    y = y[idx].astype(float)
    if cochrane_piazessi:
        reg=LinearRegression().fit(X[:-holding_period],y[holding_period:,-1:])
        X_add = np.concatenate([ reg.predict(X[:-holding_period]),np.ones((holding_period,1))],axis=0)
        X = np.concatenate([X,X_add],axis=1)
    if diebold_li or diebold_li_direct or pc_nn:
        # Compute the principal components of in sample
        n_components=3
        pca=PCA(n_components=n_components)
        X_pc=pca.fit_transform(X)
        if pc_nn:
            X = X_pc
        else:
            X_pc_pred = np.empty((len(X), n_components))
            for comp in range(n_components):
                reg=LinearRegression().fit(X_pc[:-holding_period,comp:comp+1],X_pc[holding_period:,comp])
                X_pc_pred[:,comp] = reg.predict(X_pc[:,comp:comp+1])
        
        if diebold_li_direct:
            yield_pred = pca.inverse_transform(X_pc_pred[-holding_period:])
            yields_comb = np.concatenate([X, yield_pred],axis=0)
            maturities = [2,3,4,5,7,10]
            excess_ret_comb = np.empty((len(yields_comb),len(maturities)))

            for i in range(len(maturities)):
                mat = maturities[i]
                excess_ret_comb[holding_period:,i] = (mat*yields_comb[:-holding_period,mat-1]- 
                    (mat*12-holding_period)/12*yields_comb[holding_period:,mat-int(holding_period/12)-1]-
                    holding_period/12*yields_comb[:-holding_period,int(holding_period/12)-1])
            
            excess_ret_comb=np.concatenate((excess_ret_comb, np.expand_dims(np.mean(excess_ret_comb,axis=1),axis=1)),axis=1)
                
            return excess_ret_comb[-1:,:], 0
        
        # add diebold li factors to inputs
        if diebold_li:
            X = np.concatenate([X,X_pc_pred],axis=1)
    
    learning_rate_fn = optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=0.02, decay_steps=1.0, decay_rate=0.001)
    sgd=optimizers.SGD(learning_rate=learning_rate_fn,momentum=0.9,nesterov=True)
    earlystopper=EarlyStopping(patience=20,restore_best_weights=True,min_delta=1e-6)
    fr=int(0.85*X.shape[0])
    our_ann=Sequential()
    for i in range(l):
        if i==0:
            our_ann.add(Dropout(u,input_shape=(X.shape[1],), seed = seed_))
            our_ann.add(Dense(n, activation='relu',input_dim=X.shape[1],bias_initializer='he_normal',kernel_initializer='he_normal',kernel_regularizer=regularizers.l1_l2(L)))
            #our_ann.add(Dropout(u))
        else:
            our_ann.add(Dense(n, activation='relu',bias_initializer='he_normal',kernel_initializer='he_normal',kernel_regularizer=regularizers.l1_l2(L)))
        our_ann.add(Dropout(u))   
    # our_ann.add(BatchNormalization())       
    our_ann.add(Dense(y.shape[1],activation='linear',bias_initializer='he_normal',kernel_initializer='he_normal'))
    our_ann.compile(optimizer=sgd,loss='mse')
    mm=MinMaxScaler(feature_range=(-1,1))
    
    history=our_ann.fit(mm.fit_transform(X[:fr-holding_period+1,:]),y[holding_period:fr+1,:],epochs=500,shuffle=True,validation_data=(mm.transform(X[fr:-holding_period,:]),y[fr+holding_period:,:]),callbacks=[earlystopper],verbose=0)
    return our_ann.predict(mm.transform(X[-1:,:]), verbose=0), np.min(history.history['val_loss'])

def ann_macro(X,y,m,n,l,L,u,seed_,holding_period=12):
    """
    Returns the forecast based on the ann approach

    Parameters
    ----------
    X : In sample predictors
    X_oos : Out of sample predictors
    y : In sample dependent variable
    n : number of nodes
    l : number of layers
    Returns
    -------
    Prediction of y.

    """

    
    # remove nans
    X= X.copy()
    y=y.copy()
    m = m.copy()
    idx = ~np.isnan(X).any(axis=1)
    X = X[idx].astype(float)
    y = y[idx].astype(float)
    m = m[idx].astype(float)
    fr=int(0.85*X.shape[0])

    #Scale the predictors for training
    Xscaler_train =  MinMaxScaler(feature_range=(-1,1))
    Xscaler_train.fit(X[:-holding_period,:])
    X_scaled_train = Xscaler_train.transform(X[:fr-holding_period+1,:])
    X_scaled_val = Xscaler_train.transform(X[fr:-holding_period,:])
    X_scaled_oos = Xscaler_train.transform(X[-1:,:])
    
    Xexog_scaler_train =  MinMaxScaler(feature_range=(-1,1))
    Xexog_scaler_train.fit(m[:-holding_period,:])
    Xexog_scaled_train = Xexog_scaler_train.transform(m[:fr-holding_period+1,:])
    Xexog_scaled_val = Xexog_scaler_train.transform(m[fr:-holding_period,:])
    Xexog_scaled_oos = Xexog_scaler_train.transform(m[-1:,:])

    # Keras requires 3D tuples for training.
    X_scaled_train = np.expand_dims(X_scaled_train, axis=1)
    X_scaled_val = np.expand_dims(X_scaled_val, axis=1)
    X_scaled_oos = np.expand_dims(X_scaled_oos, axis=1)
    Xexog_scaled_train = np.expand_dims(Xexog_scaled_train, axis=1)
    Xexog_scaled_val = np.expand_dims(Xexog_scaled_val, axis=1)
    Xexog_scaled_oos = np.expand_dims(Xexog_scaled_oos, axis=1)

    y = np.expand_dims(y, axis=1)
    layers=dict()
    for i in range(l + 1):
        if i == 0:
            layers["ins_main"] = Input(shape=(1, X_scaled_train.shape[2]))
        elif i == 1:
            layers["dropout" + str(i)] = Dropout(u)(layers["ins_main"])
            layers["hidden" + str(i)] = Dense(
                n,
                kernel_regularizer=regularizers.l1_l2(L),
                bias_initializer="he_normal",
                kernel_initializer="he_normal",
                activation="relu",
            )(layers["dropout" + str(i)])
        elif i > 1 & i <= n:
            layers["dropout" + str(i)] = Dropout(u)(layers["hidden" + str(i - 1)])
            layers["hidden" + str(i)] = Dense(
                n,
                kernel_regularizer=regularizers.l1_l2(L),
                bias_initializer="he_normal",
                kernel_initializer="he_normal",
                activation="relu",
            )(layers["dropout" + str(i)])

    # Model for yield variables
    layers["ins_exog"] = Input(shape=(1, Xexog_scaled_train.shape[2]))

    # Merge macro / yield networks
    layers["merge"] = Concatenate()([layers["hidden" + str(i)], layers["ins_exog"]])
    layers["dropout_final"] = Dropout(u)(layers["merge"])
    layers["BN"] = BatchNormalization()(layers["dropout_final"])
    layers["output"] = Dense(
        y.shape[2], bias_initializer="he_normal", kernel_initializer="he_normal"
    )(layers["BN"])

    model = Model(inputs=[layers["ins_main"], layers["ins_exog"]], outputs=layers["output"])

    # Compile model
    sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    earlystopper = EarlyStopping(patience=20, restore_best_weights=True, min_delta=1e-6)
    # learning_rate_fn = optimizers.schedules.InverseTimeDecay(
    #     initial_learning_rate=0.02, decay_steps=1.0, decay_rate=0.001)
    # sgd=optimizers.SGD(learning_rate=learning_rate_fn,momentum=0.9,nesterov=True)
    # earlystopper=EarlyStopping(patience=20,restore_best_weights=True,min_delta=1e-6)

    model.compile(loss="mean_squared_error", optimizer=sgd)
    history = model.fit(
        [X_scaled_train, Xexog_scaled_train],
        y[holding_period:fr+1,:],
        epochs=500,
        callbacks=[earlystopper],
        validation_data=([X_scaled_val, Xexog_scaled_val], y[fr+holding_period:,:]),
        batch_size=32,
        shuffle=True,
        verbose=0,
    )
    return model.predict([X_scaled_oos, Xexog_scaled_oos], verbose=0), np.min(history.history['val_loss'])

def multiple_ann(X,m,y,holding_period=12, **kwargs):
    random_state = kwargs.get("random_state")
    seed_ = set_seed(random_state = random_state)
    num_nns = kwargs.get('num_nns',1)
    num_best = kwargs.get('num_best',1)
    n = kwargs.get("n", 3)
    l = kwargs.get("l", 1)
    L = kwargs.get("L", 0.01)
    u = kwargs.get("u", 0.3)
    cochrane_piazessi = kwargs.get("cochrane_piazessi", False)
    diebold_li = kwargs.get("diebold_li", False)
    diebold_li_direct = kwargs.get("diebold_li_direct", False)
    pc_nn = kwargs.get("pc_nn", False)
    


    losses=[]
    for j in range(num_nns):
        if m is None:
            losses.append(ann(X=X,y=y,n=n,l=l,L=L,u=u,seed_=seed_,holding_period=holding_period, cochrane_piazessi=cochrane_piazessi, 
                               diebold_li=diebold_li,
                               diebold_li_direct=diebold_li_direct,
                               pc_nn=pc_nn))
        else:
            losses.append(ann_macro(X=X,y=y,m=m,n=n,l=l,L=L,u=u,seed_=seed_,holding_period=holding_period))
    losses.sort(key=lambda q:q[1]) 
    avg_pred=sum(i[0] for i in losses[:num_best])/num_best
    avg_loss=sum(i[1] for i in losses[:num_best])/num_best
    return avg_pred, [i[1] for i in losses]

# COMMAND ----------

from joblib import Parallel, delayed

# COMMAND ----------

def pc_regression(
    yields,
    macro,
    cpi,
    recession,
    lookback=12,
    holding_period=12,
    maturities = [24,36,48,60,84,120],
    begin_date='1971-08',
    end_date='2018-12',
    window='expanding',
    T_in=221,
    n_components=3,
    macro_components=8,
    cp_params = "default",
    cp_lookback = 120,
    short_rate_param = 0.98,
    methods_provided = None,
    num_simulations = 5000,
    ann_num_nns = 100,
    ann_num_best = 10,
    n_jobs = -1,
    seed = 0,
    ):
    # Arguments added for the local version:
    # ann_num_nns, ann_num_best: number of networks fitted / averaged (hard-coded 100 / 10 in the original)
    # n_jobs: number of local processes for the neural networks (joblib; the original used Spark)
    # seed: seed for numpy's global random generator (bootstrap simulations; the original set no seed)
    np.random.seed(seed)

    if recession is not None:
        pmi_idx = recession['PMI'] == 1
        nber_idx = recession['NBER'] ==1
    
    if cp_params is None:
        cp_params=[]
    if cp_params =="default":
        cp_params = [0.975,0.987,0.995]
    elif not isinstance(cp_params,list):
        cp_params = [cp_params]
        
    maturities = [_ for _ in maturities if _>holding_period]
    
    # get right period
    yields=yields.loc[begin_date:end_date]
    if macro is not None:
        macro=macro.loc[begin_date:end_date]  
    yields_reduced=yields[list(range(12,132,12))]
    
    
    # inflation trend
    if cpi is not None:    
        inflation = np.log(cpi.iloc[12:].to_numpy()/cpi.iloc[:-12].to_numpy())
        inflation_trend_dict=dict()
        for cp_param in cp_params:
            weight_window = (1-cp_param)*np.array([np.power(cp_param,i) for i in range(cp_lookback)])
            inf_trend = np.convolve(inflation,weight_window)[:-cp_lookback+1]
            inf_trend[:cp_lookback-1] = np.nan
            inf_trend_df = pd.DataFrame(inf_trend,index=cpi.iloc[12:].index,columns = ['inflation_trend'])
            inflation_trend_dict[str(cp_param)] = inf_trend_df.copy()
           
    # get yield changes
    yield_changes = yields_reduced.diff(periods=lookback)
    # simulated_yield_changes = simulated_yields.copy()
    # simulated_yield_changes[:,:,:] = np.nan
    # simulated_yield_changes[lookback:,:,:] = simulated_yields[lookback:,:,:]-simulated_yields[:-lookback,:,:]

    # hanson lucca wright (2021)
    hanson_predictors = pd.DataFrame(index = yields_reduced.index)
    hanson_predictors['level'] = yields_reduced.iloc[:,0].copy()
    hanson_predictors['slope'] =  yields_reduced.iloc[:,-1]-yields_reduced.iloc[:,0]
    hanson_predictors[['level_change','slope_change']] = hanson_predictors.diff(periods=lookback).copy()   

    # get excess return
    excess_ret = pd.DataFrame(index = yields.index, columns = maturities)
    
    for i in range(len(maturities)):
        mat = maturities[i]
        excess_ret.loc[excess_ret.index[holding_period:],mat] = (mat/12*yields.iloc[:-holding_period,mat-1].values - 
            (mat-holding_period)/12*yields.iloc[holding_period:,mat-holding_period-1].values-
            holding_period/12*yields.iloc[:-holding_period,holding_period-1].values)
    
    # equally weighted average
    excess_ret['EW']=excess_ret.mean(axis=1)

    

    # compute 1 year forward rates
    forwards = yields[[12]].copy()
    for i in range(24,132,12):
        forwards[i]=i/12*yields[i].copy() - (i-12)/12*yields[i-12].copy()
    forward_changes=forwards.diff(periods=lookback)
    
    # store output in dictionary
    output={}
    output['yields']=yields
    
    output['yield changes']=yield_changes
    output['forwards']=forwards
    output['forward changes']=forward_changes
    output['Excess returns']=excess_ret
    output["hanson predictors"] = hanson_predictors
    
    short_rate_trend = pd.DataFrame(index=yields.index,columns = ['short rate trend'])
    short_rate_trend.iloc[0,0] = yields_reduced.iloc[0,0]
    for update_month in range(1,len(short_rate_trend)):
        short_rate_trend.iloc[update_month] = (1-short_rate_param)*yields_reduced.iloc[update_month,0]+short_rate_param*short_rate_trend.iloc[update_month-1]
    short_rate_trend = short_rate_trend.astype(float)
    output['short rate trend'] = short_rate_trend

    # store plot of excess returns
    fig = plt.figure(dpi = 500)
    if 24 in excess_ret.columns:  # guard added: no 2-year excess return for a 24-month holding period
        plt.plot(excess_ret.index, excess_ret[24],label='2Y excess return',linestyle='--')
    plt.plot(excess_ret.index, excess_ret[120],label='10Y excess return',linestyle='-.')
    plt.ylabel('%')
    plt.legend(fontsize="x-small")
    output["Figure: excess returns"] = fig

    # to save the figure uncomment below
    # fig.savefig("excess_returns_plot.png")
    plt.close(fig)

    
    ensemble_with_cieslak_povala_methods = ['Yield changes','Yield changes + macro (revised)','Yield changes + macro (vintage)']
    
    methods = ['Yields', 'Yields without PC1','Yield changes', 'Forwards','Forwards without PC1', 'Forward changes', 'Hanson Lucca Wright',
               'Yields + macro (revised)','Yield changes + macro (revised)','Forwards + macro (revised)', 
               'Forward changes + macro (revised)', 'Yields + macro (vintage)','Yield changes + macro (vintage)',
               'Forwards + macro (vintage)', 'Forward changes + macro (vintage)','Yields + macro (ex yields, revised)',
               'Yield changes + macro (ex yields, revised)','Forwards + macro (ex yields, revised)', 
               'Forward changes + macro (ex yields, revised)', 'Yields + macro (ex yields, vintage)','Yield changes + macro (ex yields, vintage)',
               'Forwards + macro (ex yields, vintage)', 'Forward changes + macro (ex yields, vintage)', "Yields + yield changes",
               'Yields (nonoverlapping)', 
               'Yield changes (nonoverlapping)', 'Forwards (nonoverlapping)', 'Forward changes (nonoverlapping)', "Cochrane Piazessi"] + [
               f'Cieslak Povala (param {str(cp_param)})' for cp_param in cp_params] + [
               f'Cieslak Povala PC (param {str(cp_param)})' for cp_param in cp_params] + [
               f'Cieslak Povala single factor (param {str(cp_param)})' for cp_param in cp_params] + [
               f'Ens {method_1} + Cieslak Povala' for method_1 in ensemble_with_cieslak_povala_methods] + [
                "Cieslak Povala + SR trend" ]+ [
                    "NN-3-1-Yields","NN-3-1-Yield changes","NN-3-1-Forwards","NN-3-1-Forward changes",
                    "NN-32-1-Yields + macro (revised)","NN-32-1-Yield changes + macro (revised)","NN-32-1-Forwards + macro (revised)","NN-32-1-Forward changes + macro (revised)",
                    "NN-3-1-Cieslak Povala (param 0.987)","NN-3-1-Cochrane Piazessi",
                    "NN-3-1-Diebold Li","NN-3-1-Diebold Li direct", 
                    "NN-3-1-Yields PC", "NN-3-1-Yield changes PC"]
    
    method_inputs = [yields_reduced, yields_reduced, yield_changes, forwards,forwards,forward_changes,hanson_predictors, yields_reduced, 
                     yield_changes, forwards,forward_changes,yields_reduced, yield_changes, forwards,forward_changes,
                     yields_reduced, yield_changes, forwards,forward_changes,yields_reduced, yield_changes, forwards,forward_changes, yields_reduced,
                     yields_reduced, yield_changes, forwards,forward_changes, forwards] + [
                     yields_reduced for _ in range(len(cp_params))] + [
                     yields_reduced for _ in range(len(cp_params))] + [
                     yields_reduced for _ in range(len(cp_params))] + [
                     None for _ in range(len(ensemble_with_cieslak_povala_methods))] + [
                    yields_reduced]+ [
                    yields_reduced, yield_changes, forwards, forward_changes,
                    yields_reduced, yield_changes, forwards, forward_changes,
                    yields_reduced, yields_reduced,
                    yields_reduced, yields_reduced,
                    yields_reduced, yield_changes,]
    
    
    r2=pd.DataFrame(columns=excess_ret.columns)
    r2_pval = pd.DataFrame(columns=excess_ret.columns)
    r2_pmi_rec=pd.DataFrame(columns=excess_ret.columns)
    r2_pmi_rec_pval = pd.DataFrame(columns=excess_ret.columns)
    r2_nber_rec=pd.DataFrame(columns=excess_ret.columns)
    r2_nber_rec_pval = pd.DataFrame(columns=excess_ret.columns)
    RMSPE=pd.DataFrame(index=[["Benchmark"]+methods],columns=excess_ret.columns)
    u_gains_df = pd.DataFrame(index=pd.MultiIndex.from_product(
                                [methods, ['gains','p-val','t-stat']],
                                names=["Factors","stat"],
                            ),
                           columns=excess_ret.columns)
    utilities_df = pd.DataFrame(index=pd.MultiIndex.from_product(
                                [['CER','Sharpe Ratio'],['Benchmark']+methods,['stat','p-val']],
                                names=['metric',"Factors",'stat'],
                            ),
                           columns=excess_ret.columns)
                        
    # get benchmark predictions equal to historical mean
    pred = pd.DataFrame(index=excess_ret.index,columns=excess_ret.columns)
    output['CP factor correlation (TS)'] = pd.DataFrame(index=excess_ret.index,columns=['CP factor corr'])
    pred_bm=excess_ret.shift(periods=holding_period).expanding().mean()
    pred_bm.iloc[:T_in] = np.nan
    output['Benchmark prediction'] = pred_bm
    mspe_bench=pd.DataFrame(index=excess_ret.index,columns=excess_ret.columns)
    mspe_bench.iloc[T_in:]=np.square(excess_ret.iloc[T_in:].values-pred_bm.iloc[T_in:].values)
    output['Benchmark MSPE'] = mspe_bench
    output["Benchmark cum SPE"] = mspe_bench.cumsum()
    RMSPE.loc["Benchmark",:] = np.sqrt(mspe_bench.mean(axis=0).values)
    pred_2y = pd.DataFrame()

    # get benchmark utilities
    # Economic value needs the risk-free return over the holding period, taken from the one-year-spaced
    # forward rates; for holding periods that are not in forwards.columns (1, 3, 6 months) the original code
    # fails, so the economic-value part is skipped there (only R2 is reported for those settings).
    economic_value = holding_period in forwards.columns
    trading_sr_df = pd.DataFrame(columns = excess_ret.columns)
    if economic_value:
        utilities_bm, weights_bm, returns_bm = mv_final(predictions_df = pred_bm, returns_df = excess_ret, forwards_df = forwards, 
                                         holding_period = holding_period, maturities=maturities)
        output[f"Utilities: benchmark"] = utilities_bm
        output[f"Weights: benchmark"] = weights_bm
        output[f"Trading returns: benchmark"] = returns_bm
        bm_sr = returns_bm.mean()/returns_bm.std()
        trading_sr_df = pd.DataFrame(columns = excess_ret.columns)
        trading_sr_df.loc['Benchmark'] = bm_sr.to_numpy()
        cer_bm = cer(utilities_bm)
        utilities_df.loc[("CER","Benchmark",'stat')] = cer_bm.to_numpy()
        utilities_df.loc[("Sharpe Ratio","Benchmark",'stat')] = (returns_bm.mean()/returns_bm.std()).to_numpy()
        output[f'CER: benchmark'] = cer_bm
    
    # get pc predictions
    if methods_provided is None:
        numbers_to_consider = list(range(12)) + [-4,-3,-2,-1]
        numbers_to_consider = [0,2,3,5]
        numbers_to_consider = [-1]
        numbers_to_consider = [4]
    

    else:
        methods_to_consider = methods_provided
        numbers_to_consider = [methods.index(_) for _ in methods_to_consider]

    u_grid =[0.1,0.3,0.5]
    L_grid =[0.001,0.01,0.1,1]
    hyper_freq = 60
    len_oos = len(pred)-T_in
    retrain_grid = list(range(0,len_oos,hyper_freq))
    comb_grid = []
    for u in u_grid:
        for L in L_grid:
            for retrain_t in retrain_grid:
                comb_grid.append((u,L,retrain_t))
    
    best_hyperparams = {}
    best_val_losses = {}
    
    for j in numbers_to_consider:
        met=methods[j]
        val_losses = pd.DataFrame(index = excess_ret.index, columns = range(ann_num_nns))
        print(met)
        skip_later = False
        random_state = 100
        met_input=method_inputs[j]
        met_input_copy=met_input.copy()
        
        pred=pd.DataFrame(index=excess_ret.index,columns=excess_ret.columns)
        betas = {col: pd.DataFrame(index = excess_ret.index, columns = ['PC ' + str(_+1) for _ in range(n_components)]) for col in excess_ret.columns}
        # the original also sent "Yields" down this parallel route, where it would fail (missing argument);
        # as in the notebook version, "Yields" is handled by the sequential loop below
        if "NN" in met:
            if 'Cieslak Povala' in met:
                met_input_copy = met_input_copy.merge(inflation_trend_dict[met.replace("Cieslak Povala (param ","").replace(")","")[-5:]], left_index = True, right_index = True)
            # renamed from `cochrane_piazessi`: the original local variable shadowed the function of that name,
            # so the linear "Cochrane Piazessi" method failed in this version
            cochrane_piazessi_nn = "Cochrane Piazessi" in met
            diebold_li = False
            diebold_li_direct = False
            pc_nn = False
            if "Diebold Li direct" in met:
                diebold_li_direct = True
            elif "Diebold Li" in met:
                diebold_li = True
            if "PC" in met:
                pc_nn = True

            
            
            skip_later = True
            # local processes via joblib instead of Spark; inputs are passed directly instead of broadcast
            backend = "loky"
            if "macro" in met:
                macro_met_input = macro
            else:
                macro_met_input = None
            pipe = Parallel(backend=backend, n_jobs=n_jobs, verbose=10)
            
            
            if "NN" in met:
                n_ann = int(met.split("-")[1])
                l_ann = int(met.split("-")[2])
                hypertune_output = pipe(
                delayed(parallel_function)(
                    parallel_input=met_input_copy,
                    parallel_macro=macro_met_input,
                    parallel_excess_ret=excess_ret,
                    holding_period=holding_period,
                    t=T_in+retrain_t,
                    T_in=T_in,
                    window=window,
                    function = multiple_ann,
                    num_nns = ann_num_nns,
                    num_best = ann_num_best,
                    n = n_ann,
                    l = l_ann,
                    L = L_ann,
                    u = u_ann,
                    random_state = random_state,
                    cochrane_piazessi = cochrane_piazessi_nn,
                    diebold_li = diebold_li,
                    diebold_li_direct=diebold_li_direct,
                    pc_nn = pc_nn
                ) for (u_ann, L_ann, retrain_t) in comb_grid) 

                for retrain_t in retrain_grid:
                    best_val_loss = np.inf
                    for retrain_idx, (u_ann, L_ann, t_comb_grid) in enumerate(comb_grid):
                        if retrain_t!=t_comb_grid:
                            continue
                        val_losses_retrain_t = sum(hypertune_output[retrain_idx][1][:ann_num_best])
                        if val_losses_retrain_t<best_val_loss:
                            best_val_loss = val_losses_retrain_t
                            best_hyperparams[retrain_t] = {"u":u_ann,"L":L_ann}
                            best_val_losses[retrain_t] = best_val_loss
                best_hyperparams_pd = pd.DataFrame.from_dict(best_hyperparams, orient='index')
                best_val_losses_pd = pd.DataFrame.from_dict(best_val_losses, orient='index')
                output[f"{met} best hyperparams"] = best_hyperparams_pd
                output[f"{met} best val losses"] = best_val_losses_pd
                print("best hyperparams")
                print(best_hyperparams_pd)

                nn_output = pipe(
                    delayed(parallel_function)(
                        parallel_input=met_input_copy,
                        parallel_macro=macro_met_input,
                        parallel_excess_ret=excess_ret,
                        holding_period=holding_period,
                        t=t,
                        T_in=T_in,
                        window=window,
                        function = multiple_ann,
                        num_nns = ann_num_nns,
                        num_best = ann_num_best,
                        n = n_ann,
                        l = l_ann,
                        L = best_hyperparams[(t-T_in)//hyper_freq*hyper_freq]["L"],
                        u = best_hyperparams[(t-T_in)//hyper_freq*hyper_freq]["u"],
                        random_state = random_state,
                        cochrane_piazessi = cochrane_piazessi_nn,
                        diebold_li = diebold_li,
                        diebold_li_direct=diebold_li_direct,
                        # corrected: the original passed `pc = pc`, which multiple_ann ignores, so the PC
                        # transformation was only used during hyperparameter tuning, not for the forecasts
                        pc_nn = pc_nn
                    )
                    for t in range (T_in, len(pred)) 
                )
                print("Done with second parallel")
                results = [_[0] for _ in nn_output]
                val_losses_arr = np.stack([np.array(_[1]) for  _ in nn_output], axis=0)
                val_losses[T_in:] = val_losses_arr.astype(float)
                val_losses = val_losses[T_in:]
                output[f"{met} val losses"] = val_losses

            if "macro" in met:
                pred_out = np.concatenate([_[:,0,:] for  _ in results], axis=0)
            else:
                pred_out = np.concatenate([_ for  _ in results], axis=0)
            pred.iloc[T_in:] = pred_out
            
        
        
        for t in range(T_in,len(pred)):#T_in,len(pred)):
            if window=="rolling":
                t_begin = t-T_in
            elif window =="expanding":
                t_begin = 0
            
            if skip_later:
                continue
            if met.startswith('Ens') and "Cieslak Povala" in met:
                pass

            # revised macro data ex yields
            elif "+ macro (ex yields, revised)" in met:
                macro_ex_yields = pd.concat((macro.iloc[:,:77],macro.iloc[:,94:]),axis=1)  # corrected: drop the 17 interest-rate series (columns 78-94 of the Jan 2019 vintage); the original dropped 84-100, the numbering of an older FRED-MD list
                #loop over maturities
                for col in range(len(pred.columns)):
                    # add +1 because if predicting obs T with holding period=12, you can use up to and 
                    # including obs T-12, which is :T-11 in python syntax.
                    pred.iloc[t,col]=pc_macro(method_inputs[j].iloc[t_begin:t-holding_period+1].values,
                                              method_inputs[j].iloc[t-holding_period:t-holding_period+1].values,
                                              macro_ex_yields.iloc[t_begin:t-holding_period+1].values,macro_ex_yields.iloc[t-holding_period:t-holding_period+1].values,
                                              excess_ret.iloc[t_begin:t-holding_period+1,col].values,holding_period=holding_period)
            
            # revised macro data
            elif "+ macro (revised)" in met:
                
                #loop over maturities
                for col in range(len(pred.columns)):
                    # add +1 because if predicting obs T with holding period=12, you can use up to and 
                    # including obs T-12, which is :T-11 in python syntax.
                    pred.iloc[t,col]=pc_macro(method_inputs[j].iloc[t_begin:t-holding_period+1].values,
                                              method_inputs[j].iloc[t-holding_period:t-holding_period+1].values,
                                              macro.iloc[t_begin:t-holding_period+1].values,macro.iloc[t-holding_period:t-holding_period+1].values,
                                              excess_ret.iloc[t_begin:t-holding_period+1,col].values,holding_period=holding_period)
            
            ## Get vintage macro data. Read transformed macro series (contains 1m lag). The oldest vintage
            # data set (data set 1) is from 1999-08, for dates before we only implement the lag, but 
            # still have revised data. 
            elif " + macro (vintage)" in met:
                date_prediction=pred.index[t-holding_period]
                
                # count the difference in months between current point and 1999-08, the first month of vintage data
                counter=relativedelta.relativedelta(date_prediction,pd.to_datetime('1999-08')).months+12*relativedelta.relativedelta(date_prediction,pd.to_datetime('1999-08')).years
                
                # if first time
                if t==T_in:
                    macro_vintage=pd.read_csv(PATH / 'vintage data/macro_1.csv',header=None)#.values[149:,:]
                    macro_vintage.index = pd.date_range(start='4/1/1959', periods=len(macro_vintage),freq='MS')

                    # Read non-transformed macro series
                    macro_orig=pd.read_csv(PATH / 'vintage data/1999-08.csv',skiprows=1).iloc[:,1:]#,header=None)#.values[153:,1:]
                    macro_orig.index = pd.date_range(start='1/1/1959', periods=len(macro_orig),freq='MS')

                    # If original variable wasn't available last month, lag once more. 
                    for col in range(macro_vintage.shape[1]):
                        if str(macro_orig.iloc[-1,col])=='nan':
                            macro_vintage.iloc[1:,col]=macro_vintage.iloc[:-1,col].values
                
                # if beyond 1999-08
                elif counter>=1:
                    file_title = 'vintage data/macro_'+str(counter+1)+'.csv'
                    try:
                        macro_vintage=pd.read_csv(PATH / file_title,header=None)#.values[149:,:]
                    except OSError:
                        continue
                    macro_vintage.index = pd.date_range(start='4/1/1959', periods=len(macro_vintage),freq='MS')
                    file_title = 'vintage data/'+str(date_prediction.year)+'-'+str(date_prediction.month).zfill(2)+'.csv'
                    try:
                        macro_orig=pd.read_csv(PATH / file_title,skiprows=1).iloc[:,1:]
                    except OSError:
                        continue
                    macro_orig.index = pd.date_range(start='1/1/1959', periods=len(macro_orig),freq='MS')
                  
                    # If original variable wasn't available last month, take the month before that.
                    for col in range(macro_vintage.shape[1]):
                        if str(macro_orig.iloc[-1,col])=='nan':
                            macro_vintage.iloc[1:,col]=macro_vintage.iloc[:-1,col].values
                
                macro_vintage=macro_vintage.loc[begin_date:]
                #loop over maturities
                for col in range(len(pred.columns)):
                    pred.iloc[t,col]=pc_macro(method_inputs[j].iloc[t_begin:t-holding_period+1].values,method_inputs[j].iloc[t-holding_period:t-holding_period+1].values,macro_vintage.iloc[t_begin:t-holding_period+1].values,macro_vintage.iloc[t-holding_period:t-holding_period+1].values,excess_ret.iloc[t_begin:t-holding_period+1,col].values,holding_period=holding_period)
             
            # non-overlapping annual data
            elif "(nonoverlapping)" in met:
                if t%holding_period==0:
                    ann_rows = [_ for _ in range(0,t-holding_period+1,holding_period)]
                    pred.iloc[t]=pc(method_inputs[j].iloc[t_begin:t-holding_period+1].iloc[ann_rows].values,excess_ret.iloc[t_begin:t-holding_period+1].iloc[ann_rows].values,holding_period=1)
                    
            
            # just returns
            elif "without PC1" in met:
                pred.iloc[t]=pc(method_inputs[j].iloc[t_begin:t-holding_period+1].values,excess_ret.iloc[t_begin:t-holding_period+1].values,holding_period=holding_period, skip_first_components=1)
            
            elif met.startswith('Cieslak Povala single factor'):
                X_in = method_inputs[j]
                X_in = X_in.merge(inflation_trend_dict[met.replace("Cieslak Povala single factor (param ","").replace(")","")], left_index = True, right_index = True)
                pred.iloc[t], output[f'Cieslak Povala figure {t}: ct'], output[f'Cieslak Povala figure {t}: cf_t']=cieslak_povala_single_factor(X_in.iloc[t_begin:t-holding_period+1,:-1].values,X_in.iloc[t_begin:t-holding_period+1,-1:].values,excess_ret.iloc[t_begin:t-holding_period+1].values,holding_period=holding_period,maturities=maturities)
                
                
            elif met.startswith('Cieslak Povala PC'):
                X_in = method_inputs[j]
                X_in = X_in.merge(inflation_trend_dict[met.replace("Cieslak Povala PC (param ","").replace(")","")], left_index = True, right_index = True)
                pred.iloc[t]=cieslak_povala_pc(X_in.iloc[t_begin:t-holding_period+1].values,excess_ret.iloc[t_begin:t-holding_period+1].values,holding_period=holding_period)
            
            elif met.startswith('Cieslak Povala + SR trend'):
                X_in = method_inputs[j]
                X_in = X_in.merge(inflation_trend_dict['0.987'], left_index = True, right_index = True)
                X_in = X_in.merge(short_rate_trend, left_index = True, right_index = True)
                pred.iloc[t]=direct_regression(X_in.iloc[t_begin:t-holding_period+1].values,excess_ret.iloc[t_begin:t-holding_period+1].values,holding_period=holding_period)
     
            elif met.startswith('Cieslak Povala'):
                X_in = method_inputs[j]
                X_in = X_in.merge(inflation_trend_dict[met.replace("Cieslak Povala (param ","").replace(")","")], left_index = True, right_index = True)
                pred.iloc[t]=direct_regression(X_in.iloc[t_begin:t-holding_period+1].values,excess_ret.iloc[t_begin:t-holding_period+1].values,holding_period=holding_period)
            
            elif met.startswith('Hanson'):
                pred.iloc[t]=direct_regression(method_inputs[j].iloc[t_begin:t-holding_period+1].values,
                                               excess_ret.iloc[t_begin:t-holding_period+1].values,holding_period=holding_period)
            
            elif met.startswith("Cochrane Piazessi"):
                pred.iloc[t],cp_factor=cochrane_piazessi(method_inputs[j].iloc[t_begin:t-holding_period+1].values,excess_ret.iloc[t_begin:t-holding_period+1].values,holding_period=holding_period,return_cp_factor_only = True)
                df_copi = pd.DataFrame(index = method_inputs[j].index, columns = ['CP factor'])
                df_copi.iloc[-len(cp_factor):]=cp_factor
                df_copi['Lagged CP factor'] = df_copi['CP factor'].shift(12)
                output['CP factor correlation (TS)'].iloc[t] = df_copi.astype(float).corr().iloc[1,0]

            elif met=="Yields + yield changes":
                pred.iloc[t]=pc_double(method_inputs[j].iloc[t_begin:t-holding_period+1].values,yield_changes.iloc[t_begin:t-holding_period+1].values,excess_ret.iloc[t_begin:t-holding_period+1].values,holding_period=holding_period)
                
                
                
            else:
                pred.iloc[t]=pc(method_inputs[j].iloc[t_begin:t-holding_period+1].values,excess_ret.iloc[t_begin:t-holding_period+1].values,holding_period=holding_period)
                for col in range(len(pred.columns)):
                    colname = list(pred.columns)[col]
                    coef_outp =pc_coef(method_inputs[j].iloc[t_begin:t-holding_period+1].values,excess_ret.iloc[t_begin:t-holding_period+1, col].values,holding_period=holding_period)
                    betas[colname].iloc[t] = np.copy(coef_outp[0][1:])

        # obtain utilities
        # print(f"pred.iloc[T_in:][[120]]: {pred.iloc[T_in:][[120]]}")
        # print(f"excess_ret.iloc[:T_in][[120]]: {excess_ret.iloc[:T_in][[120]]}")
        # print(f"excess_ret.iloc[T_in+holding_period:][[120]]: {excess_ret.iloc[T_in+holding_period:][[120]]}")
        # print(f"forwards.iloc[T_in:][12]: {forwards.iloc[T_in:][12]}")

        # prediction utilities
        # utilities, _ = mv_def(pred.iloc[T_in:][[120]].to_numpy(), excess_ret.iloc[:T_in][[120]].to_numpy(), excess_ret.iloc[T_in:][[120]].to_numpy(), 
        #                forwards.iloc[T_in:][12].to_numpy())
        # non-overlapping forecasts exist only once a year, so the comparison with the monthly benchmark
        # utilities fails; only R2 is reported for them
        if economic_value and "(nonoverlapping)" not in met:
            utilities, weights, trading_returns = mv_final(predictions_df = pred, returns_df = excess_ret, forwards_df = forwards, 
                                         holding_period = holding_period, maturities=maturities)
            output[f"Utilities: {met}"] = utilities
            output[f"Weights: {met}"] = weights
            output[f"Trading returns: {met}"] = trading_returns
            trading_sr_df.loc[met] = trading_returns.mean()/trading_returns.std()
        
            cer_out = cer(utilities)
            output[f'CER: {met}'] = cer_out

            # utility gains
  
            c_gains=100*(cer_out-cer_bm)
            u_gains_df.loc[(met, "gains")] = c_gains.to_numpy()
            utilities_df.loc[("CER",met,'stat')] = cer_out.to_numpy()
            utilities_df.loc[("Sharpe Ratio",met,'stat')] = (trading_returns.mean()/trading_returns.std()).to_numpy()
            for colname in u_gains_df.columns:
            
                if u_gains_df.loc[(met,"gains"),colname]<0:
                    continue
                reg=sm.OLS(utilities[colname].to_numpy()-utilities_bm[colname].to_numpy(),np.ones(utilities.shape[0]))
                res=reg.fit(cov_type='HAC',cov_kwds={'maxlags':max(holding_period-1,1)})
                u_alpha=res.params[0]
                var=res.cov_params()[-1,-1]
                u_sig=np.sqrt(var)
                #u_sig=res.bse[0]
                #u_alpha[i,j],u_t[i,j]=cer_sig(u[i],u[j])
                u_t=u_alpha/u_sig
                u_p=2*tstat.cdf(-abs(u_t), df = utilities.shape[0] - 1)

            
                u_gains_df.loc[(met, "p-val"),colname] = u_p
                utilities_df.loc[("CER", met,'p-val'),colname] = u_p
                u_gains_df.loc[(met, "t-stat"),colname] = u_t
            
            

        if met.startswith('Ens') and "Cieslak Povala" in met:
            method_1 = met.replace('Ens ','').replace(' + Cieslak Povala', '')
            pred = 0.5*(output[method_1+' prediction'] + output['Cieslak Povala (param 0.987) prediction'])
            
        for key, val in betas.items():
            val = val.astype(float)
            if not np.isnan(val).all(axis=None):
                if key!="EW":
                    key = str(int(int(key)/12))+"Y"
                    
                title = f"Regression coefficients over time ({met}, {key})"
                    
                output[title] = val.copy()
                fig = plt.figure(dpi = 200)
                plt.plot(val)
                plt.legend(labels=val.columns)
                plt.title(title)
                output[f'Figure: {title}'] = fig
                plt.close(fig)

        # coefficients for hanson lucca wright
        if met.startswith('Hanson'):
            for mat_col, maturity in enumerate(maturities+["EW"]):
                for hanson_subsample in ["pre-2000","post-2000"]:
                    hanson_settings = [1,2,3,4,5]
                    cof = pd.DataFrame(index=pd.MultiIndex.from_product(
                                        [hanson_settings, ['coef','p-value','t-stat']],
                                        names=["setting","statistic"]),
                                        columns = ['const']+list(hanson_predictors.columns)+["R2","adj-R2"])
                                
                    for hanson_setting in hanson_settings:
                        if hanson_setting>=4:
                            hanson_num_predictors = 4
                            hanson_first_predictor = hanson_setting-3
                        else:
                            hanson_num_predictors = hanson_setting+1
                            hanson_first_predictor = 0
                        if hanson_subsample=="pre-2000":
                            
                            if holding_period==12:
                                end_date = "2001-01-01"
                            else:
                                end_date = "2000-07-01"
                            pars, se, t_score, p_vals, r_squared, adj_r_squared =direct_regression_coef(
                            method_inputs[j].loc[:end_date].values[:,hanson_first_predictor:hanson_num_predictors],
                                                           excess_ret.loc[:end_date].values[:,mat_col],holding_period=holding_period)
                        else:
                            # note, to follow hanson et al, drop periods inbetween (perhaps not best, because we're not dropping other obs)
                            if holding_period==12:
                                start_date = "2001-01-01"
                            else:
                                start_date = "2000-07-01"
                            pars, se, t_score, p_vals, r_squared, adj_r_squared =direct_regression_coef(
                                method_inputs[j].loc[start_date:].values[:,hanson_first_predictor:hanson_num_predictors],
                                                           excess_ret.loc[start_date:].values[:,mat_col],holding_period=holding_period)
                        for comp in range(len(pars)): 
                            if hanson_setting>=4 and comp>0:
                                comp_to_use = comp+hanson_first_predictor
                            else:
                                comp_to_use = comp
                            cof.loc[(hanson_setting,'coef'),cof.columns[comp_to_use]] = pars[comp]
                            cof.loc[(hanson_setting,'p-value'),cof.columns[comp_to_use]] = p_vals[comp]
                            cof.loc[(hanson_setting,'t-stat'),cof.columns[comp_to_use]] = t_score[comp]
                        cof.loc[(hanson_setting,'coef'),"R2"] = r_squared
                        cof.loc[(hanson_setting,'coef'),"adj-R2"] = adj_r_squared
                    #cof.columns = ['const']+list(hanson_predictors.columns)+["R2","adj-R2"]

                    if maturity=="EW":
                        between_brackets = "EW"
                    else:
                        between_brackets = str(int(maturity/12))+"Y"
                    output[met + f': {hanson_subsample} regression ({between_brackets})'] = cof.T
            
        # get coefficients of a full sample regression
        cof = pd.DataFrame(index=pd.MultiIndex.from_product(
                                [list(excess_ret.columns), ['coef','p-value','p-value (BH)','t-stat']],
                                names=["maturity","statistic"],
                            ),
                           columns=pd.Index([_ for _ in range(n_components+1)]+["R2"],name = met+' PCs')
                        )
        
        macro_pcs=pd.DataFrame()
        if "macro" in met:
            t_vals_arr = np.empty((len(pred.columns),n_components+1+macro_components))
            t_vals_arr[:,:] = np.nan
            simulated_t_vals = np.empty((len(pred.columns),n_components+1+macro_components,num_simulations))
            simulated_t_vals[:,:,:] = np.nan
        else:
            t_vals_arr = np.empty((len(pred.columns),n_components+1))
            t_vals_arr[:,:] = np.nan
            simulated_t_vals = np.empty((len(pred.columns),n_components+1,num_simulations))
            simulated_t_vals[:,:,:] = np.nan
        
        
        for col in range(len(pred.columns)):
            if met=="Cochrane Piazessi" and col==0:   
                cp_factor = cochrane_piazessi(method_inputs[j].values,excess_ret.values,holding_period=holding_period, return_cp_factor = True)
                df_copi = pd.DataFrame(index = method_inputs[j].index, columns = ['CP factor'])
                df_copi.iloc[-len(cp_factor):]=cp_factor
                df_copi['Lagged CP factor'] = df_copi['CP factor'].shift(12)
                output['CP factor'] = df_copi
                output['CP factor correlations'] = df_copi.astype(float).corr()
                
            if any([method_substring in met for method_substring in ['nonoverlapping','Cieslak Povala', "Cochrane Piazessi"]]):
                pars = None

            elif "+ macro (ex yields, revised)" in met:
                macro_ex_yields = pd.concat((macro.iloc[:,:77],macro.iloc[:,94:]),axis=1)  # corrected: drop the 17 interest-rate series (columns 78-94 of the Jan 2019 vintage); the original dropped 84-100, the numbering of an older FRED-MD list
                pars, se, t_score, p_vals, r_squared, sig_pcs_used, pcs, all_macro_pcs, yield_pcs, macro_loadings = pc_coef_macro(
                    method_inputs[j].values[holding_period:-holding_period],macro_ex_yields.values[holding_period:-holding_period],
                    excess_ret.iloc[:,col].values[holding_period:-holding_period],holding_period=holding_period)
                output[met + ': macro PC loadings'] = macro_loadings
                output[met + ': all macro pcs'] = all_macro_pcs

                
            elif "+ macro (revised)" in met:
                # pars, se, t_score, p_vals, r_squared, sig_pcs_used, pcs, all_macro_pcs, yield_pcs, macro_loadings = pc_coef_macro(
                #     method_inputs[j].values[holding_period:-holding_period],macro.values[holding_period:-holding_period],
                #     excess_ret.iloc[:,col].values[holding_period:-holding_period],holding_period=holding_period)
                pars, se, t_score, p_vals, r_squared, sig_pcs_used, pcs, all_macro_pcs, yield_pcs, macro_loadings = pc_coef_macro(
                    method_inputs[j].values,macro.values,
                    excess_ret.iloc[:,col].values,holding_period=holding_period)
                _,_, t_score_non_hac, _, _, _, _, _, _, _ = pc_coef_macro(
                    method_inputs[j].values,macro.values,
                    excess_ret.iloc[:,col].values,holding_period=holding_period)
                output[met + ': macro PC loadings'] = macro_loadings
                output[met + ': all macro pcs'] = all_macro_pcs
                
                if "Yields + macro (revised)"==met:
                    simulated_yields, simulated_other_predictors = simulate_yields(yields_reduced.to_numpy(), pcs, n_components=n_components, 
                                                                                    num_simulations=num_simulations)
                    simulated_excess_ret = np.empty(shape=(len(yields),len(maturities),num_simulations))
                    simulated_excess_ret[:,:,:] = np.nan
                    for i in range(len(maturities)):
                        mat = maturities[i]                
                        simulated_excess_ret[holding_period:,i,:] = (mat/12*simulated_yields[:-holding_period,int(mat/12)-1,:] - 
                            (mat-holding_period)/12*simulated_yields[holding_period:, int((mat-holding_period)/12)-1,:]-
                            holding_period/12*simulated_yields[:-holding_period, int(holding_period/12)-1,:])
                    
                    # equally weighted average
                    simulated_excess_ret = np.concatenate((simulated_excess_ret,np.expand_dims(np.mean(simulated_excess_ret,axis=1),axis=1)),axis=1)
                    output['simulated yields'] = simulated_yields
                    output[f"simulated predictors for {met}"] = simulated_other_predictors
    
                    
    
                    col_idx = list(range(n_components+1))+[_+n_components+1 for _ in sig_pcs_used]
                    t_vals_arr[col,col_idx] = t_score_non_hac
                    for n_simul in range(num_simulations):
                        # if met=="Yields":
                        #     method_inputs_sim = simulated_yields[:,:,n_simul]
                        # else:
                        #     method_inputs_sim = simulated_yield_changes[:,:,n_simul]
                        
                        _,_,simul_t_score,_,_,_,_ = pc_coef(simulated_other_predictors[:,:,n_simul],simulated_excess_ret[:,col,n_simul],
                            holding_period=holding_period, take_pc = False, robust_se = False)
                        # _,_,simul_t_score,_,_,_,_ = pc_coef(method_inputs_sim,simulated_excess_ret[:,col,n_simul],
                        #     holding_period=holding_period, take_pc = True)
                        simulated_t_vals[col,col_idx,n_simul] = simul_t_score

            
            elif "+ macro (vintage)" in met:
                try:
                    pars, se, t_score, p_vals, r_squared, sig_pcs_used, pcs, all_macro_pcs, yield_pcs, macro_loadings = pc_coef_macro(
                        method_inputs[j].values[holding_period:-holding_period],macro_vintage.values[holding_period:],
                        excess_ret.iloc[:,col].values[holding_period:-holding_period],holding_period=holding_period)
                    output[met + ': macro PC loadings'] = macro_loadings
                except:
                    continue
            else:
                pars, se, t_score, p_vals, r_squared, pcs, loadings= pc_coef(method_inputs[j].values,excess_ret.iloc[:,col].values,holding_period=holding_period)
                _, _, t_score_non_hac, _, _, _, _= pc_coef(method_inputs[j].values,excess_ret.iloc[:,col].values,holding_period=holding_period, robust_se=False)
                if met in []:
                    # simulated_yields, simulated_other_predictors = simulate_yields(yields_reduced.to_numpy(), pcs, n_components=n_components, 
                    #                                                             num_simulations=num_simulations)
                    simulated_excess_ret, simulated_other_predictors =simulate_excess_returns(excess_ret.iloc[:,col:col+1].values, pcs, 
                        n_components=n_components, num_simulations=num_simulations)
                    
                    output[f"simulated predictors for {met}"] = simulated_other_predictors
                    
                    # to set simulated data to real data for testing
                    # simulated_yields[:,:,0] = yields_reduced
                    
                    # simulated_yield_changes = simulated_yields.copy()
                    # simulated_yield_changes[:,:,:]= np.nan
                    # simulated_yield_changes[holding_period:,:,:] = simulated_yields[holding_period:,:,:]-simulated_yields[:-holding_period,:,:]
                    

                    # simulated_excess_ret = np.empty(shape=(len(yields),len(maturities),num_simulations))
                    # simulated_excess_ret[:,:,:] = np.nan
                    # for i in range(len(maturities)):
                    #     mat = maturities[i]                
                    #     simulated_excess_ret[holding_period:,i,:] = (mat/12*simulated_yields[:-holding_period,int(mat/12)-1,:] - 
                    #         (mat-holding_period)/12*simulated_yields[holding_period:, int((mat-holding_period)/12)-1,:]-
                    #         holding_period/12*simulated_yields[:-holding_period, int(holding_period/12)-1,:])
                
                    # to set simulated data to real data for testing
                    # simulated_excess_ret[:,:,0] =excess_ret.to_numpy()
                    
                    # equally weighted average
                    # simulated_excess_ret = np.concatenate((simulated_excess_ret,np.expand_dims(np.mean(simulated_excess_ret,axis=1),axis=1)),axis=1)
                    # output['simulated yields'] = simulated_yields
                    output[f'simulated excess returns for {met}']=simulated_excess_ret

                    t_vals_arr[col,:] = t_score_non_hac
                    for n_simul in range(num_simulations):
                        # if met=="Yields":
                        #     method_inputs_sim = simulated_yields[:,:,n_simul]
                        # else:
                        #     method_inputs_sim = simulated_yield_changes[:,:,n_simul]
                        
                        _,_,simul_t_score,_,_,_,_ = pc_coef(simulated_other_predictors[:,:,n_simul],simulated_excess_ret[:,:,n_simul],
                            holding_period=holding_period, take_pc = False, robust_se = False)
                        # _,_,simul_t_score,_,_,_,_ = pc_coef(method_inputs_sim,simulated_excess_ret[:,col,n_simul],
                        #     holding_period=holding_period, take_pc = True)
                        simulated_t_vals[col,:,n_simul] = simul_t_score

                
                # store principal components
                df_pcs = pd.DataFrame(index = yield_changes.index, columns = ['PC ' + str(_+1) for _ in range(n_components)])
                df_pcs.iloc[-len(pcs):]=pcs
                for col_ in list(df_pcs.columns):
                    df_pcs['Lagged '+ col_] = df_pcs[col_].shift(12)
                output[met + ': PCs'] = df_pcs
                output[met + ': PC correlations'] = df_pcs.astype(float).corr()
                output[met + ': PC loadings'] = loadings       
            if pars is not None:
                for comp in range(len(pars)): 
                    if comp<4:
                        col_name=comp
                    else:
                        col_name='Macro PC '+str(sig_pcs_used[comp-4]+1)
                        macro_pcs[col_name] = pcs[:,comp-1]
                    cof.loc[(pred.columns[col],'coef'),col_name] = pars[comp]
                    cof.loc[(pred.columns[col],'p-value'),col_name] = p_vals[comp]
                    cof.loc[(pred.columns[col],'t-stat'),col_name] = t_score[comp]
                cof.loc[(pred.columns[col],'coef'),"R2"] = r_squared


            
        cof = cof.rename({0:"const"},axis=1)
        new_index = list(cof.columns)
        new_index.remove("R2")
        new_index=new_index[:4]+sorted(new_index[4:])
        cof = cof.reindex(columns=new_index + ["R2"])
        if met in ["Yields + macro (revised)"]:
            nonnan_col_idx=~np.isnan(t_vals_arr).all(axis=0)
            higher_t_stat_arr = (np.abs(simulated_t_vals)>np.abs(t_vals_arr[:,:,np.newaxis])-0.0001).astype(float).mean(axis=-1)
            
            cof.loc[(slice(None),'p-value (BH)'),:cof.columns[-2]] = higher_t_stat_arr[:,nonnan_col_idx]
            output[f"{met}: simulated t-vals"] = simulated_t_vals[:,nonnan_col_idx,:]
            output[f"{met}: actual t-vals"] = t_vals_arr[:,nonnan_col_idx]
        
        output[met + ': full sample regression'] = cof
        
        # Full sample macro PCs
        if "macro" in met:
            
            df_pcs = pd.DataFrame(index = yield_changes.index,columns= new_index[1+n_components:])
            for col in macro_pcs.columns:
                df_pcs.loc[:,col].iloc[-len(pcs):] = macro_pcs[col]
            output[met + ': PCs'] = df_pcs
        
        output[met+' prediction'] = pred
        
    
        mspe=pd.DataFrame(index=excess_ret.index,columns=excess_ret.columns)
        mspe.iloc[T_in:]=np.square(excess_ret.iloc[T_in:].values-pred.iloc[T_in:].values)
        output[met+' MSPE'] = mspe
        output[met + " cum SPE"] = mspe.cumsum()
        

        for i in r2.columns:
            try:
                idx = ~np.isnan(mspe[i].astype(float).values)
                this_mspe = mspe[i].loc[idx]
                this_mspe_bench = mspe_bench[i].loc[idx]
                r2.loc[met,i]=1-np.nansum(this_mspe.values)/np.nansum(this_mspe_bench.values)
                if recession is not None:
                    r2_nber_rec.loc[met,i]=1-np.nansum(this_mspe.loc[nber_idx].values)/np.nansum(this_mspe_bench.loc[nber_idx].values)
                    r2_pmi_rec.loc[met,i]=1-np.nansum(this_mspe.loc[pmi_idx].values)/np.nansum(this_mspe_bench.loc[pmi_idx].values)
                if r2.loc[met,i]>0:
                    r2_pval.loc[met,i] = R2_pval(excess_ret[i].loc[idx].values,output['Benchmark prediction'][i].loc[idx].values,pred[i].loc[idx].values, holding_period=holding_period)
                else:
                    r2_pval.loc[met,i] = np.nan
                if "(nonoverlapping)" not in met:
                    RMSPE.loc[met,i] = np.sqrt(np.nanmean(mspe[i].astype(float).values))
            except Exception as e:
                continue
 
    # correlations between predictions
    pred_2y = pd.DataFrame()
    pred_10y = pd.DataFrame()
    for met in methods_to_consider:
        if 24 in output[met+' prediction'].columns:  # guard added, as above
            pred_2y[met] = output[met+' prediction'][24]
        pred_10y[met] = output[met+' prediction'][120]
    output['Prediction correlations 2Y'] = pred_2y.astype("float").corr()
    output['Prediction correlations 10Y'] = pred_10y.astype("float").corr()
      
    output['R2']=r2
    output['R2 (NBER recession)'] = r2_nber_rec
    output['R2 (PMI recession)'] = r2_pmi_rec
    output['R2_pval'] = r2_pval
    output["RMSPE"] = RMSPE
    output[f"CER gains"] = u_gains_df.dropna(how="all")
    output["Utilities"] = utilities_df.dropna(how="all")
    output["Trading SR"] = trading_sr_df.dropna(how="all")
    
    if holding_period==12 and lookback==12:
        if np.all([met in methods_to_consider for met in ['Yields','Benchmark', 'Yield changes']]):
            # Cumulative squared forecast errors
            cum_spe_24m = pd.DataFrame()
            cum_spe_120m =pd.DataFrame()
            for met in ['Yields','Benchmark', 'Yield changes']:
                if met+ " cum SPE" in output.keys():
                    cum_spe_24m[met] = output[met + " cum SPE"][24]
                    cum_spe_120m[met] = output[met + " cum SPE"][120]
            output["Cum SPE 24m"] = cum_spe_24m
            output["Cum SPE 120m"] = cum_spe_120m

            for mat in ["24", "120"]:
                idx = ~output["Cum SPE "+mat+"m"].isna().all(axis=1)
                fig = plt.figure(dpi = 200)
                plt.plot(output["Cum SPE "+mat+"m"].loc[idx])
                plt.legend(labels=output["Cum SPE "+mat+"m"].columns)
                plt.title(f"Cumulative Squared Prediction Error ({str(int(mat)//12)}Y)")
                output["Figure cum SPE "+mat+"m"] = fig           
                plt.close(fig)

                # in recessions (NBER)
                fig, ax = plt.subplots()
                cum_mspe = output['Cum SPE '+mat+"m"].copy()
                cum_mspe['NBER'] = recession['NBER']
                idx = ~cum_mspe.isna().any(axis=1)
                to_plot = cum_mspe.loc[idx]
                s = to_plot.iloc[:,:3].to_numpy()
                ax.plot(s)
                ax.set(title = "Cumulative Squared Prediction Error ("+mat+"m)")
                ax.fill_between(range(len(s)), np.min(s), np.max(s), where=(to_plot.iloc[:,-1]==1), alpha=0.5, color="grey")
                labels = [''] +[str(year) for year in range(1990,2022,4)]
                #ax.set_xticklabels([''] + [str(list(to_plot.loc[idx].index)[dt])[:4] for dt in list(range(0,350,50))]+['2018'])
                ax.set_xticklabels(labels)
                plt.legend(labels=output["Cum SPE "+mat+"m"].columns)
                output["Figure cum SPE "+mat+"m (NBER recession)"] = fig
                plt.close(fig)

                fig, ax = plt.subplots()
                s = to_plot.iloc[:,0].to_numpy() - to_plot.iloc[:,2].to_numpy()
                ax.plot(s)
                ax.set(title = "Cumulative Squared Prediction Error (Yields - yield changes, "+mat+"m)")
                ax.fill_between(range(len(s)), np.min(s), np.max(s), where=(to_plot.iloc[:,-1]==1), alpha=0.5, color="grey")
                ax.set_xticklabels([''] + [str(list(to_plot.loc[idx].index)[dt])[:4] for dt in list(range(0,350,50))])
                output["Figure cum SPE "+mat+"m (yields-yield changes, NBER recession)"] = fig
                plt.close(fig)

                fig, ax = plt.subplots()
                s = to_plot.iloc[:,1].to_numpy() - to_plot.iloc[:,2].to_numpy()
                ax.plot(s)
                ax.set(title = "Cumulative Squared Prediction Error (benchmark - yield changes, "+mat+"m)")
                ax.fill_between(range(len(s)), np.min(s), np.max(s), where=(to_plot.iloc[:,-1]==1), alpha=0.5, color="grey")
                ax.set_xticklabels([''] + [str(list(to_plot.loc[idx].index)[dt])[:4] for dt in list(range(0,350,50))])
                output["Figure cum SPE "+mat+"m (benchmark-yield changes, NBER recession)"] = fig
                plt.close(fig)

                fig, ax = plt.subplots()
                s = to_plot.iloc[:,0].to_numpy() - to_plot.iloc[:,1].to_numpy()
                ax.plot(s)
                ax.set(title = "Cumulative Squared Prediction Error (yields - benchmark, "+mat+"m)")
                ax.fill_between(range(len(s)), np.min(s), np.max(s), where=(to_plot.iloc[:,-1]==1), alpha=0.5, color="grey")
                ax.set_xticklabels([''] + [str(list(to_plot.loc[idx].index)[dt])[:4] for dt in list(range(0,350,50))])
                output["Figure cum SPE "+mat+"m (yields - benchmark, NBER recession)"] = fig
                plt.close(fig)

                # in recessions (PMI)
                fig, ax = plt.subplots()
                cum_mspe = output['Cum SPE '+mat+"m"].copy()
                cum_mspe['PMI'] = recession['PMI']
                idx = ~cum_mspe.isna().any(axis=1)
                to_plot = cum_mspe.loc[idx]
                s = to_plot.iloc[:,:3].to_numpy()
                ax.plot(s)
                ax.set(title = f"Cumulative Squared Prediction Error ({str(int(mat)//12)}Y)")
                ax.fill_between(range(len(s)), np.min(s), np.max(s), where=(to_plot.iloc[:,-1]==1), alpha=0.5, color="grey")
                #ax.set_xticklabels([''] + [str(list(to_plot.loc[idx].index)[dt])[:4] for dt in list(range(0,350,50))])
                ax.set_xticklabels(labels)
                plt.legend(labels=output["Cum SPE "+mat+"m"].columns)
                output["Figure cum SPE "+mat+"m (PMI recession)"] = fig
                plt.close(fig)

                fig, ax = plt.subplots()
                s = to_plot.iloc[:,0].to_numpy() - to_plot.iloc[:,2].to_numpy()
                ax.plot(s)
                ax.set(title = "Cumulative Squared Prediction Error (Yields - yield changes, "+mat+"m)")
                ax.fill_between(range(len(s)), np.min(s), np.max(s), where=(to_plot.iloc[:,-1]==1), alpha=0.5, color="grey")
                ax.set_xticklabels([''] + [str(list(to_plot.loc[idx].index)[dt])[:4] for dt in list(range(0,350,50))])
                output["Figure cum SPE "+mat+"m (yields-yield changes, PMI recession)"] = fig
                plt.close(fig)

                fig, ax = plt.subplots()
                s = to_plot.iloc[:,1].to_numpy() - to_plot.iloc[:,2].to_numpy()
                ax.plot(s)
                ax.set(title = "Cumulative Squared Prediction Error (benchmark - yield changes, "+mat+"m)")
                ax.fill_between(range(len(s)), np.min(s), np.max(s), where=(to_plot.iloc[:,-1]==1), alpha=0.5, color="grey")
                ax.set_xticklabels([''] + [str(list(to_plot.loc[idx].index)[dt])[:4] for dt in list(range(0,350,50))])
                output["Figure cum SPE "+mat+"m (benchmark-yield changes, PMI recession)"] = fig
                plt.close(fig)

                fig, ax = plt.subplots()
                s = to_plot.iloc[:,0].to_numpy() - to_plot.iloc[:,1].to_numpy()
                ax.plot(s)
                ax.set(title = "Cumulative Squared Prediction Error (yields - benchmark, "+mat+"m)")
                ax.fill_between(range(len(s)), np.min(s), np.max(s), where=(to_plot.iloc[:,-1]==1), alpha=0.5, color="grey")
                ax.set_xticklabels([''] + [str(list(to_plot.loc[idx].index)[dt])[:4] for dt in list(range(0,350,50))])
                output["Figure cum SPE "+mat+"m (yields - benchmark, PMI recession)"] = fig
                plt.close(fig)

            

        # plot PCs
        for met in ["Yields","Yield changes","Yields + macro (revised)"]:
            if met+": PCs" in output.keys():
                cols = [_ for _ in output[met + ': PCs'].columns if "Lagged" not in _]
                idx = ~output[met + ': PCs'].isna().all(axis=1)
                fig = plt.figure(dpi=500,figsize=(15,4.8))
                plt.plot(output[met + ': PCs'].loc[idx,cols])
                plt.legend(labels=output[met + ': PCs'][cols].columns)
                output[met + ': PCs figure'] = fig
                plt.close(fig)

        # plot change in pcs vs pcs of change
        if "Yields: PCs" in output.keys() and "Yield changes: PCs" in output.keys():
            for comp_nr in range(n_components):
                fig = plt.figure(dpi=500,figsize=(15,4.8))
                change_in_yields_plot_df = output["Yield changes: PCs"].iloc[:,comp_nr]
                idx = ~change_in_yields_plot_df.isna()
                plt.plot(change_in_yields_plot_df.loc[idx])
                # astype(float) added: with pandas 1.5 the object-dtype frame (None after shift) cannot be differenced
                yields_plot_df = output["Yields: PCs"].astype(float).diff(periods=lookback).iloc[:,comp_nr]
                idx = ~yields_plot_df.isna()
                plt.plot(yields_plot_df.loc[idx])
                plt.legend(labels = [f"PC {comp_nr+1} of changes in yields", f"Change in PC {comp_nr+1} of yields"])
                output[f"Figure: pcs of changes vs changes in pcs {comp_nr+1}"] = fig
                # plt.savefig(f"Figure - pcs of changes vs changes in pcs {comp_nr+1}.png")
                plt.close(fig)
                
                
        # store principal components
        if "Yields" in methods_to_consider and "Yield changes" in methods_to_consider and "Yields + macro (revised)" in methods_to_consider:
            df_pcs = pd.DataFrame(index = yields.index, columns = ['Yield PC ' + str(_+1) for _ in range(n_components)]+ ['Yield change PC ' + str(_+1) for _ in range(n_components)]+['Macro PC ' + str(_+1) for _ in range(macro_components)])
            df_pcs.iloc[-len(output['Yields: PCs']):,:n_components] = output['Yields: PCs'][["PC 1", "PC 2", "PC 3"]].values
            df_pcs.iloc[-len(output['Yield changes: PCs']):,n_components:2*n_components] = output['Yield changes: PCs'][["PC 1", "PC 2", "PC 3"]].values
            df_pcs.iloc[-len(output["Yields + macro (revised): all macro pcs"]):,2*n_components:]=output["Yields + macro (revised): all macro pcs"]
            output['Yields and macro PC correlations'] = df_pcs.astype(float).corr()
        if "Yields: PCs" in output.keys() and "Yield changes: PCs" in output.keys():
            df_pcs = pd.DataFrame(index = yields.index, 
                columns = ['PC of yields' + str(_+1) for _ in range(n_components)]+ 
                ['Yield change PC ' + str(_+1) for _ in range(n_components)]+['Change in PC of yields' + str(_+1) for _ in range(n_components)])
            df_pcs[['Yield change PC ' + str(_+1) for _ in range(n_components)]] = output['Yield changes: PCs'][["PC 1", "PC 2", "PC 3"]]
            df_pcs[['PC of yields' + str(_+1) for _ in range(n_components)]]= output['Yields: PCs'][["PC 1", "PC 2", "PC 3"]]
            df_pcs[['Change in PC of yields' + str(_+1) for _ in range(n_components)]] = output['Yields: PCs'][["PC 1", "PC 2", "PC 3"]].diff(periods=lookback)
            output['Yields and change in yields PC correlations'] = df_pcs.astype(float).corr()
            
        # plot predictions
        if "Forwards" in methods_to_consider:
            idx = ~output["Forwards prediction"].isna().all(axis=1)
            fig = plt.figure(dpi = 500,figsize=(10,6))
            plt.plot(output["Forwards prediction"].loc[idx,24])
            plt.plot(output["Benchmark prediction"].loc[idx,24])
            plt.plot(output["Excess returns"].loc[idx,24])
            plt.legend(labels=['PCR prediction', 'Benchmark prediction', 'Realised returns'])
            plt.title("2Y excess returns")
            output["Figure 2Y excess returns"] = fig
            plt.close(fig)
            
        # plot predictions
        if "Cieslak Povala (param 0.987)" in methods_to_consider:
            idx = ~output["Cieslak Povala (param 0.987) prediction"].isna().all(axis=1)
            for mat in [24,120]:
                fig = plt.figure(dpi = 500,figsize=(10,6))
                plt.plot(output["Cieslak Povala (param 0.987) prediction"].loc[idx,mat])
                plt.plot(output["Benchmark prediction"].loc[idx,mat])
                plt.plot(output["Excess returns"].loc[idx,mat])
                plt.legend(labels=['Cieslak Povala prediction', 'Benchmark prediction', 'Realised returns'])
                plt.title(f"{str(int(mat/12))}Y excess returns")
                output[f"Figure {str(int(mat/12))}Y excess returns (Cieslak Povala)"] = fig
                plt.close(fig)     
        # plot predictions
        if "Cieslak Povala + SR trend" in methods_to_consider:
            idx = ~output["Cieslak Povala + SR trend prediction"].isna().all(axis=1)
            for mat in [24,120]:
                fig = plt.figure(dpi = 500,figsize=(10,6))
                plt.plot(output["Cieslak Povala + SR trend prediction"].loc[idx,mat])
                plt.plot(output["Benchmark prediction"].loc[idx,mat])
                plt.plot(output["Excess returns"].loc[idx,mat])
                plt.legend(labels=['Cieslak Povala prediction', 'Benchmark prediction', 'Realised returns'])
                plt.title(f"{str(int(mat/12))}Y excess returns")
                output[f"Figure {str(int(mat/12))}Y excess returns (Cieslak Povala + SR trend prediction)"] = fig
                plt.close(fig)      
    
    # loadings of PCs
    if "Yields: PCs" in output.keys() and "Yield changes: PCs" in output.keys():
        fig = plt.figure(dpi = 500)
        plt.plot(range(1,11),output['Yields: PC loadings'][:,0],label='Yields: PC 1',linestyle='--')
        plt.plot(range(1,11),output['Yields: PC loadings'][:,1],label='Yields: PC 2',linestyle='-.')
        plt.plot(range(1,11),output['Yields: PC loadings'][:,2],label='Yields: PC 3',linestyle=':')
        plt.plot(range(1,11),output['Yield changes: PC loadings'][:,0],label='Changes in yields: PC 1',color='blue')
        plt.plot(range(1,11),output['Yield changes: PC loadings'][:,1],label='Changes in yields: PC 2',color='orange')
        plt.plot(range(1,11),output['Yield changes: PC loadings'][:,2],label='Changes in yields: PC 3',color='green')
        plt.xlabel('Maturity')
        plt.ylabel('Loading')
        plt.legend(fontsize="x-small")
        output["Figure PC loadings yields"] = fig
        #plt.savefig(f"Figure - PC loadings yields.png")
        plt.close(fig)
    
    return output
