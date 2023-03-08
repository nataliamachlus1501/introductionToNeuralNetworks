# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 12:12:40 2023

@author: natal
"""

#bliblioteki

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
KATALOG_PROJEKTU = os.path.join(os.getcwd(),"projekt_2")
KATALOG_WYKRESOW = os.path.join(KATALOG_PROJEKTU, "wykresy")
os.makedirs(KATALOG_WYKRESOW, exist_ok=True)

#%%wczytanie tabel utorzonych w ramach projektu 1

tabele = []

#z oknem o maksymalnym odchyleniu RR
tabela_gl = pd.read_excel('tabela_bazowa.xlsx')

tabela_gl = tabela_gl.drop('nazwa_pliku', axis = 1)

tabele.append(tabela_gl)

#z oknem o maksymalnym sr RR
tabela_sr = pd.read_excel('tabela_sr.xlsx')

#nie będziemy potrzebować tej kolumny, usuńmy ją
tabela_sr = tabela_sr.drop('nazwa_pliku', axis = 1)

tabele.append(tabela_sr)

#z oknem o maksymalnym odchyleniu RR
tabela_std = pd.read_excel('tabela_std.xlsx')

tabela_std = tabela_std.drop('nazwa_pliku', axis = 1)

tabele.append(tabela_std)

#%%Podstawowe informacje o tabelach

nazwy_tabel = ['Tabela bazowa','Tabela z oknem o największym srednim RR',
               'Tabela z oknem o największym odchyleniu RR']

for i in range(len(tabele)):
    print('\n'+nazwy_tabel[i]+'\n')
    print(tabele[i].info())
        
#W obu tabelach istnieje kolumna nienumeryczna plec

#%%Ogólna informacja kompletnosci danych

for i in range(len(tabele)):
    print('\n'+nazwy_tabel[i]+'\n')
    print("ogólna informacja o kompletnosci danych")
    print('\n-->brak danych w danej kolumnie:\n')
    print(tabele[i].isnull().sum())

#Wszystkie dane są kompletne

#%%Podział na dane uczące i testujące

train_datas = []
X_train = []
y_train = []
X_test = []
y_test = []

from sklearn.model_selection import StratifiedShuffleSplit

#przeprowadzimy próbkowanie warstwowe
for i in range(len(tabele)):
    print('\n'+nazwy_tabel[i]+'\n')
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
    for train_index, test_index in split.split(tabele[i],tabele[i]['dekada']):
        strat_train_set = tabele[i].loc[train_index]
        strat_test_set = tabele[i].loc[test_index]
    
    print('proporcje kategorii dekad w zbiorze testowym:')
    print(strat_test_set['dekada'].value_counts()/len(strat_test_set))

    print('\nproporcje kategorii dekad w całym zbiorze danych:')
    print(tabele[i]['dekada'].value_counts()/len(tabele[i]))
    
    #rozdzielenie czynników prognostycznych od etykiet
    #trenujace
    X_t = strat_train_set.drop('dekada', axis = 1)
    y_t = strat_train_set['dekada'].copy()
    X_train.append(X_t)
    y_train.append(y_t)
    
    #testujace
    X_tst = strat_test_set.drop('dekada', axis = 1)
    y_tst= strat_test_set['dekada'].copy()
    X_test.append(X_tst)
    y_test.append(y_tst)
    
    #kopia danych uczacych(do wizualizacji)
    train_data = strat_train_set.copy()
    train_datas.append(train_data)


#%%Poznanie i wizualizacja danych trenujących 

#zamiana danych nienumerycznych(kolumna plec) na numeryczne
from sklearn.preprocessing import OrdinalEncoder

for i in range(len(tabele)):
    print('\n'+nazwy_tabel[i]+'\n')
    train_encoder = OrdinalEncoder()
    train_datas[i]['plec'] = train_encoder.fit_transform(train_datas[i][['plec']])
    # f -> 0, m -> 1
    #macierz korelacji dla danych numerycznych
    corr_matrix = train_datas[i].corr()
    print(corr_matrix['dekada'].sort_values(ascending = False))
    print('\n')
    

#Tabela bazowa:
#korelacja dodatnia
#P(0da)        0.616356
#P(da0)        0.588164
#P(0d)         0.504261
#P(a0d)        0.488516

#korelacja ujemna
#P(aa)        -0.505568
#P(add)       -0.510121
#P(dd)        -0.513815
#P(aad)       -0.535416

#Tabela z oknem o największym srednim RR
#korelacja dodatnia
#P(0)_sr               0.473104
#P(0d0)_sr             0.427858
#P(0d)_sr              0.425083
#P(a0)_sr              0.415975

#korelacja ujemna
#SDNN_okno_sr         -0.453619
#RMSSD_okno_sr        -0.467352
#pNN20_okno_sr        -0.474625
#pNN50_okno_sr        -0.521026

#Tabela z oknem o największym odchyleniu RR
#korelacja dodatnia
#P(0da)_od             0.336926
#min_RR_okno_od        0.281385
#P(ada)_od             0.271852
#P(dad)_od             0.267926

#korelacja ujemna
#P(aa)_od             -0.304076
#max_RR_okno_od       -0.342872
#pNN50_okno_od        -0.372879
#SDNN_okno_od         -0.589894

#%%wykresy korelacji najbardziej istotnych zmiennych

from pandas.plotting import scatter_matrix

#Tabela bazowa
attribs_gl = ['dekada','P(0da)','P(da0)','P(0d)','P(aad)']
scatter_matrix(train_datas[0][attribs_gl], figsize = (10,8))
plt.savefig(os.path.join(KATALOG_WYKRESOW,'wykres_korelacji_tabela_gl.jpg'), dpi=300)

#Tabela z oknem o największym srednim RR
attribs_sr = ['dekada','P(0)_sr','P(0d0)_sr','P(0d)_sr','P(a0)_sr']
scatter_matrix(train_datas[1][attribs_sr], figsize = (10,8))
plt.savefig(os.path.join(KATALOG_WYKRESOW,'wykres_korelacji_tabela_sr.jpg'), dpi=300)
                         
#Tabela z oknem o największym odchyleniu RR
attribs_std = ['dekada','P(0da)_od','min_RR_okno_od','P(ada)_od','P(dad)_od']
scatter_matrix(train_datas[2][attribs_std], figsize = (10,8))
plt.savefig(os.path.join(KATALOG_WYKRESOW,'wykres_korelacji_tabela_std.jpg'), dpi=300)

#%%Przygotowywanie danych pod algorytmy uczenia maszynowego: krok 1 - zamiana danych nienumerycznych

#zamiana danych nienumerycznych dla danych uczących bez etykiet
encoders = []

encoder_gl = OrdinalEncoder()
X_train[0]['plec'] = encoder_gl.fit_transform(X_train[0][['plec']])
encoders.append(encoder_gl)


#Tabela z oknem o największym srednim RR
encoder_sr = OrdinalEncoder()
X_train[1]['plec'] = encoder_sr.fit_transform(X_train[1][['plec']])
encoders.append(encoder_sr)

#Tabela z oknem o największym odchyleniu RR
encoder_std = OrdinalEncoder()
X_train[2]['plec'] = encoder_std.fit_transform(X_train[2][['plec']])
encoders.append(encoder_std)

# f -> 0, m -> 1
#%%Przygotowywanie danych pod algorytmy uczenia maszynowego: krok 2 - skalowanie
from sklearn.preprocessing import StandardScaler

#skalowanie danych uczących bez etykiet
X_train_scaled = []

#Tabela bazowa
scaler_gl = StandardScaler()

scaled_features_gl = scaler_gl.fit_transform(X_train[0].values)
X_train_gl_scaled = pd.DataFrame(scaled_features_gl, index = X_train[0].index, columns = X_train[0].columns)
X_train_scaled.append(X_train_gl_scaled)


#Tabela z oknem o największym srednim RR
scaler_sr = StandardScaler()

scaled_features_sr = scaler_sr.fit_transform(X_train[1].values)
X_train_sr_scaled = pd.DataFrame(scaled_features_sr, index = X_train[1].index, columns = X_train[1].columns)
X_train_scaled.append(X_train_sr_scaled)


#Tabela z oknem o największym odchyleniu RR
scaler_std = StandardScaler()

scaled_features_std = scaler_std.fit_transform(X_train[2].values)
X_train_std_scaled = pd.DataFrame(scaled_features_std, index = X_train[2].index, columns = X_train[2].columns)
X_train_scaled.append(X_train_std_scaled)

#%%Wizualizacja PCA 
#PCA stosujemy na przeskalowanych danych uczących
#explained_variance_ratio_ = explained_variance_ / np.sum(explained_variance_)

from sklearn.decomposition import PCA

for i in range(len(tabele)):
    print('\n'+nazwy_tabel[i]+'\n')
    pca = PCA(random_state=42)
    data_reduced = pca.fit_transform(X_train_scaled[i])
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    #1 wykres - wyjasniony procent wariancji
    plt.figure(figsize=(6, 4))
    plt.plot(cumsum)
    plt.title('Wariancja wyjasniona jako funkcja liczby wymiarów')
    plt.ylabel('wyjaniona wariancja')
    plt.xlabel('wymiary')
    plt.grid()
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'wykres1_PCA_tabela{}.jpg'.format(i+1)), dpi=300)

    #2 wykres - wartosci wlasne macierzy kowariancji dla danej skladowej
    plt.figure(figsize=(6, 4))
    plt.bar(
        range(1,len(pca.explained_variance_)+1),
        pca.explained_variance_
        )
    plt.xlim([0,30])
    plt.xlabel('PCA Feature')
    plt.ylabel('Explained variance')
    plt.title('Feature Explained Variance')
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'wykres2_PCA_tabela{}.jpg'.format(i+1)), dpi=300)
    plt.show()

#%%Redukcja i wizualizacja do 2 wymiarów
plt.style.use('default')
tabele_pca2 = []

for i in range(len(tabele)):
    
    pca_2 = PCA(n_components = 2, random_state = 42)
    X_train_transformed = pca_2.fit_transform(X_train_scaled[i])

    pca2_tabela = pd.DataFrame(X_train_transformed,
                               index = X_train_scaled[i].index, columns = ['PC1', 'PC2'])

    pca2_tabela['target'] = y_train[i]
    tabele_pca2.append(pca2_tabela)

    ex_variance=np.var(X_train_transformed,axis=0)
    ex_variance_ratio = ex_variance/np.sum(ex_variance)

    Xax = X_train_transformed[:,0]
    Yax = X_train_transformed[:,1]
    cdict={20:'red',30:'green',40:'blue',50:'yellow',60:'magenta',70:'orange',80:'brown'}

    fig,ax = plt.subplots(figsize=(6,5))
    fig.patch.set_facecolor('white')
    for l in np.unique(y_train[i]):
        ix=np.where(y_train[i]==l)
        ax.scatter(Xax[ix], Yax[ix], c=cdict[l], s=40, label = l )

    ax.set_xlabel('PCA1, ' +  str(round(pca_2.explained_variance_ratio_[0]*100,2)) + '% Explained', fontsize=8)
    ax.set_ylabel('PCA2, ' +  str(round(pca_2.explained_variance_ratio_[1]*100,2)) + '% Explained', fontsize=8)


    ax.legend(fontsize = 'x-small', loc='best', markerscale=1)
    plt.title(nazwy_tabel[i])
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'PCA_2_tabela_{}.jpg'.format(i+1)), dpi=300 ) 
    plt.show()
    

#%%Redukcja i wizualizacja do 3 wymiarów
 
tabele_pca3 = []

for i in range(len(tabele)):
    pca_3 = PCA(n_components=3, random_state = 42)
    X_train_transformed = pca_3.fit_transform(X_train_scaled[i])

    pca3_tabela = pd.DataFrame(X_train_transformed,
                               index = X_train_scaled[i].index, columns =['PC1', 'PC2', 'PC3'])

    pca3_tabela['target'] = y_train[i]
    tabele_pca3.append(pca3_tabela)
    
    #wizualizacja PCA dla 3 komponentów
    ex_variance=np.var(X_train_transformed,axis=0)
    ex_variance_ratio = ex_variance/np.sum(ex_variance)

    Xax = X_train_transformed[:,0]
    Yax = X_train_transformed[:,1]
    Zax = X_train_transformed[:,2]

    cdict={20:'red',30:'green',40:'blue',50:'yellow',60:'magenta',70:'orange',80:'brown'}

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
 
    fig.patch.set_facecolor('white')
    for l in np.unique(y_train[i]):
        ix=np.where(y_train[i]==l)
        ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40, label=l)

    ax.set_xlabel('PCA1, ' +  str(round(pca_3.explained_variance_ratio_[0]*100,2)) + '% Explained', fontsize=7)
    ax.set_ylabel('PCA2, ' +  str(round(pca_3.explained_variance_ratio_[1]*100,2)) + '% Explained', fontsize=7)
    ax.set_zlabel('PCA3, ' +  str(round(pca_3.explained_variance_ratio_[2]*100,2)) + '% Explained', fontsize=7)

    ax.legend(fontsize = 'x-small', loc='best', markerscale=1)
    plt.title(nazwy_tabel[i])
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'PCA_3_tabela_{}.jpg'.format(i+1)), dpi=300 ) 
    plt.show()

#%%Redukcja do tylu wymiarów aby wyjasnic 95% wariancji dla tabeli bazowej
X_train_transformed = []

pca_95_gl = PCA(n_components = 0.95, random_state = 42)
X_train_transformed_gl = pca_95_gl.fit_transform(X_train_gl_scaled)
print(pca_95_gl.explained_variance_ratio_)

X_train_transformed.append(X_train_transformed_gl)

pca95_tabela_gl = pd.DataFrame(X_train_transformed_gl,
                           index = X_train_gl_scaled.index)
 
pca95_tabela_gl_targets = pca95_tabela_gl.copy()
pca95_tabela_gl_targets['target'] = y_train[0]

#%%Redukcja do tylu wymiarów aby wyjasnic 95% wariancji dla tabeli o max sr RR

pca_95_sr = PCA(n_components = 0.95, random_state = 42)
X_train_transformed_sr = pca_95_sr.fit_transform(X_train_sr_scaled)
print(pca_95_sr.explained_variance_ratio_)

X_train_transformed.append(X_train_transformed_sr)

pca95_tabela_sr = pd.DataFrame(X_train_transformed_sr,
                           index = X_train_sr_scaled.index)
 
pca95_tabela_sr_targets = pca95_tabela_sr.copy()
pca95_tabela_sr_targets['target'] = y_train[1]

#%%Redukcja do tylu wymiarów aby wyjasnic 95% wariancji dla tabeli o max odchyleniu RR

pca_95_std = PCA(n_components = 0.95, random_state = 42)
X_train_transformed_std = pca_95_std.fit_transform(X_train_std_scaled)
print(pca_95_std.explained_variance_ratio_)

X_train_transformed.append(X_train_transformed_std)

pca95_tabela_std = pd.DataFrame(X_train_transformed_std,
                           index = X_train_std_scaled.index)
 
pca95_tabela_std_targets = pca95_tabela_std.copy()
pca95_tabela_std_targets['target'] = y_train[2]


#%% 1 MODEL- DLA TABELI BAZOWEJ
#%%startowa próbka regresji: cały zbiór trenujący, algorytm przy domylnych ustawieniach
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

sgd_reg = SGDRegressor()
#uczymy model
sgd_reg.fit(X_train_transformed_gl, y_train[0])

#błąd dla calego zbioru uczacego regresji liniowej
sgd_reg_predictions = sgd_reg.predict(X_train_transformed_gl)
sgd_reg_mse = mean_squared_error(y_train[0], sgd_reg_predictions)
sgd_reg_rmse = np.sqrt(sgd_reg_mse)
print('błąd RMSE dla całego zbioru uczącego: ')
print(sgd_reg_rmse)

#%%sprawdzanie krzyżowe i porównanie SGDRegressor z innymi modelami

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor()
forest_reg = RandomForestRegressor()
sgd_reg = SGDRegressor()

modele = [lin_reg,tree_reg,forest_reg,sgd_reg]

def display_scores(model):
    scores = cross_val_score(model, X_train_transformed_gl, y_train[0], cv=4, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print("Wyniki: ", rmse_scores)
    print("Srednia: ", rmse_scores.mean())
    print("Odchylenie standardowe: ", rmse_scores.std())
    print('\n')

#porownanie wyników walidacji krzyżowej dla kilku różnych modeli
for model in modele:
    print(model)
    display_scores(model)
    
#%% WYSZUKIWANIE NAJLEPSZYCH WARTOŚCI HIPERPARAMETROW

from sklearn.model_selection import GridSearchCV


params = {
    'learning_rate':['constant','optimal','invscaling'],
    'max_iter':[1000,2000,4000],
    'eta0':[0.001,0.01,0.1],
    'random_state':[42],
    'alpha' : [0.001, 0.01, 0.1],
    'penalty' :['l2','l1']
        }

sgd_reg = SGDRegressor()

grid_search = GridSearchCV(sgd_reg, param_grid = params, cv = 4, scoring ='neg_mean_squared_error',
                     return_train_score = True)

grid_search.fit(X_train_transformed_gl, y_train[0])

cv_res = grid_search.cv_results_
for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    print(np.sqrt(-mean_score), params)
    

#%%Ostateczne uczenie najlepszym modelem

final_model_gl = grid_search.best_estimator_

final_model_param_gl = final_model_gl.get_params()
final_training_gl = final_model_gl.fit(X_train_transformed_gl,y_train[0])

#%%przygotowanie i testowanie danych testujacych
bledy_modeli = []

#zamiana danych nienumerycznych
X_test[0]['plec'] = encoder_gl.transform(X_test[0][['plec']])

#skalowanie
scaled_features_gl_test = scaler_gl.transform(X_test[0].values)
X_test_gl_scaled = pd.DataFrame(scaled_features_gl_test, index = X_test[0].index, columns = X_test[0].columns)

#redukcja wymiarów
X_test_transformed_gl = pca_95_gl.transform(X_test_gl_scaled) 

#Testowanie na danych uczacych
final_predictions_gl = final_training_gl.predict(X_test_transformed_gl)
final_mse_gl = mean_squared_error(y_test[0],final_predictions_gl)
final_rmse_gl = np.sqrt(final_mse_gl)

print('Błąd RMSE dla zbioru testującego: ')
print(final_rmse_gl)

bledy_modeli.append(final_rmse_gl)

#%% 2 MODEL - DLA TABELI Z OKNEM O NAJWIĘKSZYM SREDNIM RR
#%%startowa próbka regresji: cały zbiór trenujący, algorytm przy domylnych ustawieniach

sgd_reg = SGDRegressor()
#uczymy model
sgd_reg.fit(X_train_transformed_sr, y_train[1])

#błąd dla calego zbioru uczacego regresji liniowej
sgd_reg_predictions = sgd_reg.predict(X_train_transformed_sr)
sgd_reg_mse = mean_squared_error(y_train[1], sgd_reg_predictions)
sgd_reg_rmse = np.sqrt(sgd_reg_mse)
print('błąd RMSE dla całego zbioru uczącego: ')
print(sgd_reg_rmse)

#%%#%%sprawdzanie krzyżowe i porównanie SGDRegressor z innymi modelami

lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor()
forest_reg = RandomForestRegressor()
sgd_reg = SGDRegressor()

modele = [lin_reg,tree_reg,forest_reg,sgd_reg]

def display_scores(model):
    scores = cross_val_score(model, X_train_transformed_sr, y_train[1], cv=4, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print("Wyniki: ", rmse_scores)
    print("Srednia: ", rmse_scores.mean())
    print("Odchylenie standardowe: ", rmse_scores.std())
    print('\n')

for model in modele:
    print(model)
    display_scores(model)
    
#%% WYSZUKIWANIE NAJLEPSZYCH WARTOŚCI HIPERPARAMETROW

params = {
    'learning_rate':['constant','optimal','invscaling'],
    'max_iter':[3000,5000,7000],
    'random_state':[42],
    'eta0':[0.001,0.01,0.1],
    'alpha' : [0.0001,0.001, 0.01, 0.1],
    'penalty' :['l2','l1']
        }

sgd_reg = SGDRegressor()

grid_search = GridSearchCV(sgd_reg, param_grid = params, cv = 4, scoring ='neg_mean_squared_error',
                     return_train_score = True)

grid_search.fit(X_train_transformed_sr, y_train[1])

cv_res = grid_search.cv_results_
for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    print(np.sqrt(-mean_score), params)
    

#%%Ostateczne uczenie najlepszym modelem

final_model_sr = grid_search.best_estimator_

final_model_param_sr = final_model_sr.get_params()
final_training_sr = final_model_sr.fit(X_train_transformed_sr,y_train[1])

#%%przygotowanie i testowanie danych testujacych

#zamiana danych nienumerycznych
X_test[1]['plec'] = encoder_sr.transform(X_test[1][['plec']])

#skalowanie
scaled_features_sr_test = scaler_sr.transform(X_test[1].values)
X_test_sr_scaled = pd.DataFrame(scaled_features_sr_test, index = X_test[1].index, columns = X_test[1].columns)

#redukcja wymiarów
X_test_transformed_sr = pca_95_sr.transform(X_test_sr_scaled) 

#Testowanie na danych uczacych
final_predictions_sr = final_training_sr.predict(X_test_transformed_sr)
final_mse_sr = mean_squared_error(y_test[1],final_predictions_sr)
final_rmse_sr = np.sqrt(final_mse_sr)

print('Błąd RMSE dla zbioru testującego: ')
print(final_rmse_sr)

bledy_modeli.append(final_rmse_sr)

#%% 3 MODEL - DLA TABELI Z OKNEM O NAJWIĘKSZYM ODCHYLENIU RR
#%%startowa próbka regresji: cały zbiór trenujący, algorytm przy domylnych ustawieniach

sgd_reg = SGDRegressor()
#uczymy model
sgd_reg.fit(X_train_transformed_std, y_train[2])

#błąd dla calego zbioru uczacego regresji liniowej
sgd_reg_predictions = sgd_reg.predict(X_train_transformed_std)
sgd_reg_mse = mean_squared_error(y_train[2], sgd_reg_predictions)
sgd_reg_rmse = np.sqrt(sgd_reg_mse)
print('błąd RMSE dla całego zbioru uczącego: ')
print(sgd_reg_rmse)

#%%sprawdzanie krzyżowe i porównanie SGDRegressor z innymi modelami

lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor()
forest_reg = RandomForestRegressor()
sgd_reg = SGDRegressor()

modele = [lin_reg,tree_reg,forest_reg,sgd_reg]

def display_scores(model):
    scores = cross_val_score(model, X_train_transformed_std, y_train[2], cv=4, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print("Wyniki: ", rmse_scores)
    print("Srednia: ", rmse_scores.mean())
    print("Odchylenie standardowe: ", rmse_scores.std())
    print('\n')

for model in modele:
    print(model)
    display_scores(model)
    
#%% WYSZUKIWANIE NAJLEPSZYCH WARTOŚCI HIPERPARAMETROW

params = {
    'learning_rate':['constant','optimal','invscaling'],
    'max_iter':[2000,3000,5000,7000],
    'eta0':[0.001,0.01,0.1],
    'random_state':[42],
    'alpha' : [0.0001,0.001, 0.01, 0.1],
    'penalty' :['l2','l1']
        }

sgd_reg = SGDRegressor()

grid_search = GridSearchCV(sgd_reg, param_grid = params, cv = 4, scoring ='neg_mean_squared_error',
                     return_train_score = True)

grid_search.fit(X_train_transformed_std, y_train[2])

cv_res = grid_search.cv_results_
for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    print(np.sqrt(-mean_score), params)
    

#%%Ostateczne uczenie najlepszym modelem

final_model_std = grid_search.best_estimator_

final_model_param_std = final_model_std.get_params()
final_training_std = final_model_std.fit(X_train_transformed_std,y_train[2])

#%%przygotowanie i testowanie danych testujacych

#zamiana danych nienumerycznych
X_test[2]['plec'] = encoder_std.transform(X_test[2][['plec']])

#skalowanie
scaled_features_std_test = scaler_std.transform(X_test[2].values)
X_test_std_scaled = pd.DataFrame(scaled_features_std_test, index = X_test[2].index, columns = X_test[2].columns)

#redukcja wymiarów
X_test_transformed_std = pca_95_std.transform(X_test_std_scaled) 

#Testowanie na danych uczacych
final_predictions_std = final_training_std.predict(X_test_transformed_std)
final_mse_std = mean_squared_error(y_test[2],final_predictions_std)
final_rmse_std = np.sqrt(final_mse_std)

print('Błąd RMSE dla zbioru testującego: ')
print(final_rmse_std)

bledy_modeli.append(final_rmse_std)





