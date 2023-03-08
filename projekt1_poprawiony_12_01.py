# -*- coding: utf-8 -*-
"""
October 2022
    serie czasowe
"""

#%%Wstęp:

#Uaktualniony został jedynie kod niezbędny do utworzenia tabel do drugiego projektu - jest on odkomentowany,
#,natomiast kod używany jedynie do zadań z projektu 1 został zakomentowany


#%%


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ustawiamy katalogi pracy
import os
KATALOG_PROJEKTU = os.path.join(os.getcwd(),"rytm_serca")
KATALOG_DANYCH = os.path.join(KATALOG_PROJEKTU,"dane")
KATALOG_WYKRESOW = os.path.join(KATALOG_PROJEKTU, "wykresy")
os.makedirs(KATALOG_WYKRESOW, exist_ok=True)
os.makedirs(KATALOG_DANYCH, exist_ok=True)


SKAD_POBIERAC = ['./healthy_decades/', './HTX_LVH/',  './hypertension_RR_SBP/',
                 './hypertension/']

#wczytywanie danych                
czytamy = SKAD_POBIERAC[0]
print ('\nPrzetwarzamy katalog', czytamy)
pliki = os.listdir (czytamy)
[print( pliki.index(item) , ':', item)  for item in pliki]
#mamy 180 plików tekstowych zawierające dane do analizy    


def load_serie(skad , co, ile_pomin = 0, kolumny =['Interval', 'Num_contraction']):
    csv_path = os.path.join(skad, co )
    #print ( skad, co)
    seria = pd.read_csv(csv_path, sep='\t', header =None,
                        skiprows= ile_pomin, names= kolumny)
    if skad == SKAD_POBIERAC[2]:
        seria = pd.read_csv(csv_path, sep='\t',  decimal=',' )
    return seria
    
#%%podział na pliki dla poszczególnych grup wiekowych i płci
pliki_20_f = []
pliki_30_f = []
pliki_40_f = []
pliki_50_f = []
pliki_60_f = []
pliki_70_f = []
pliki_80_f = []

pliki_20_m = []
pliki_30_m = []
pliki_40_m = []
pliki_50_m = []
pliki_60_m = []
pliki_70_m = []
pliki_80_m = []

for item in pliki:
    if item.startswith('f20'):
        pliki_20_f.append(item)
        
    elif item.startswith('f30'):
        pliki_30_f.append(item)
    
    elif item.startswith('f40'):
        pliki_40_f.append(item)
        
    elif item.startswith('f50'):
        pliki_50_f.append(item)
    
    elif item.startswith('f60'):
        pliki_60_f.append(item)
    
    elif item.startswith('f70'):
        pliki_70_f.append(item)
        
    elif item.startswith('f80'):
        pliki_80_f.append(item)
        
for item in pliki:
    if item.startswith('m20'):
        pliki_20_m.append(item)
        
    elif item.startswith('m30'):
        pliki_30_m.append(item)
    
    elif item.startswith('m40'):
        pliki_40_m.append(item)
        
    elif item.startswith('m50'):
        pliki_50_m.append(item)
    
    elif item.startswith('m60'):
        pliki_60_m.append(item)
    
    elif item.startswith('m70'):
        pliki_70_m.append(item)
        
    elif item.startswith('m80'):
        pliki_80_m.append(item)
        
#%%
pliki_20 = []
pliki_30 = []
pliki_40 = []
pliki_50 = []
pliki_60 = []
pliki_70 = []
pliki_80 = []

for item in pliki:
    if item.startswith('f20') or item.startswith('m20'):
        pliki_20.append(item)
    
    elif item.startswith('f30') or item.startswith('m30'):
        pliki_30.append(item)
        
    elif item.startswith('f40') or item.startswith('m40'):
        pliki_40.append(item)
    
    elif item.startswith('f50') or item.startswith('m50'):
        pliki_50.append(item)
    
    elif item.startswith('f60') or item.startswith('m60'):
        pliki_60.append(item)
        
    elif item.startswith('f70') or item.startswith('m70'):
        pliki_70.append(item)
        
    elif item.startswith('f80') or item.startswith('m80'):
        pliki_80.append(item)
        
#%%KOMPLETNOSC SERII
#Sprawdzanie kompletnosci każdej serii:
    
# for i in range(0, len(pliki)):
#     plik = pliki[i]
#     seria = load_serie(skad = czytamy, co= plik )
#     if seria.isnull().sum().sum() != 0:
#         print('\n-->brak danych w danej kolumnie:\n') 
#         print(seria.isnull().sum())
#     else:
#         pass
# print("Wszystkie dane komletne.")

#%%sprawdzenie czy kolumna Num_contraction jest posortowana

# for i in range(0, len(pliki)):
#     plik = pliki[i]
#     seria = load_serie(skad = czytamy, co= plik )
#     if seria.Num_contraction.is_monotonic_increasing:
#         pass
#     else:
#         print('Kolumna nie jest posortowana w pliku {}'.format(plik))
# print('Dla wszystkich plików kolumny Num_contracion są posortowane rosnąco.')
 
#%%Sprawdzenie czy liczby w kolumnie Num_contraction są kolejne

# for i in range(0,len(pliki)):
#     plik = pliki[i]
#     seria = load_serie(skad = czytamy, co= plik )
#     numery = seria['Num_contraction'].values.tolist()
#     for i in range(0, len(numery)-1):
#         if numery[i+1] - numery[i] != 1:
#             print('Nie są kolejne w pliku: {a} przy numerze: {b}'.format(a=plik, b = numery[i]))
#         else:
#             pass

#%%PODSTAWOWE STATYSTYKI(dla wybranej przez użytkownika serii)

# while (True): 
#     try:
#         num_plik = int(input('Podaj indeks pliku, dla którego chciałbys zobaczyc podstawowe statystyki(0-181): '))
#         print('Podstawowe statystyki dla pliku: ', pliki[num_plik])
#         plik=pliki[num_plik]
#         seria = load_serie(skad = czytamy, co= plik )
#         print(seria.describe())
#     except  ValueError: 
#         print('Błędny numer pliku. ')
#     else:
#         break  
    
#%%WIZUALIZACJA SERII
#1)HISTOGRAMY
#histogram dla każdej serii

# for i in range(0, len(pliki)):
#     plik = pliki[i]
#     seria = load_serie(skad = czytamy, co= plik )
#     seria['Interval'].hist(bins=50, figsize=(9,6))
#     plt.title("histogramy wartosći " + plik)
#     plt.xlabel('czas między kolejnymi skurczami serca')
#     plt.ylabel('liczba skurczów')
#     plt.savefig(os.path.join(KATALOG_WYKRESOW,'histogramy_RR '+plik+'.jpg'), dpi=300 ) 
#     plt.show()

#%%histogramy dla poszczególnych dekad
# pliki_lst=[pliki_20,pliki_30,pliki_40,pliki_50,pliki_60,pliki_70,pliki_80]

# for i in range(0,len(pliki_lst)):
#     dane_dekady = []
#     for plik in pliki_lst[i]:
#         seria = load_serie(skad = czytamy, co= plik )
#         dane_dekady.append(seria) 
#     final_df = pd.concat(dane_dekady,ignore_index=True)
#     final_df['Interval'].hist(bins = 50, figsize=(9,6))
#     plt.title("histogram wartosći dla {}0 latków".format(i+2))
#     plt.xlabel('czas między kolejnymi skurczami serca')
#     plt.ylabel('liczba skurczów')
#     plt.savefig(os.path.join(KATALOG_WYKRESOW,'histogramy_RR_{}0_latków.jpg'.format(i+2)), dpi=300 ) 
#     plt.show()
    
#%%
#2)WYKRESY ZALEŻNOSCI OD CZASU
#wykres skurczów dla każdej serii(zależnosc od czasu)

# for i in range(0, len(pliki)):
#     plik = pliki[i]
#     seria = load_serie(skad = czytamy, co= plik )
#     seria["Interval"].plot(title='Interval ' + plik, figsize=(9,6))
#     plt.xlabel('numer porządkowy skurczu')
#     plt.ylabel('czas między kolejnymi skurczami serca')
#     plt.savefig(os.path.join(KATALOG_WYKRESOW, 'Interval'+ '_healthy '+plik+'.jpg'), dpi=300 ) 
#     plt.show()
    
#%%
#3)WYKRESY POINCARE
#definujemy funkcje rysujaca wykres poincare

# def wykres_poincare(rr):
    
#     rr_n = rr[:-1] #od pierwszego elementu do przedostatniego
#     rr_n1 = rr[1:] #od drugiego elementu do ostatniego

#     min_rr = np.min(rr)
#     max_rr = np.max(rr)
    
#     plt.figure(figsize=(11, 9))

#     #na osi X rr_n a na osi Y rr_n1
#     sns.scatterplot(x=rr_n, y=rr_n1, color = "#7D9EC0")
#     plt.plot(rr_n,rr_n,'k')

#     plt.xlabel(r'$RR_n (ms)$')
#     plt.ylabel(r'$RR_{n+1} (ms)$')

#     plt.text(min_rr + 20, max_rr - 50, "zwolnienie serca", fontsize=15, color="green")
#     plt.text(max_rr - 360, min_rr + 70, "przyspieszenie serca", fontsize=15, color="red")
    
    
#%%wykres poincare dla każdej serii

# for i in range(0, len(pliki)):
#     plik = pliki[i]
#     seria = load_serie(skad = czytamy, co= plik )
#     wykres_poincare(seria['Interval'].values)
#     plt.title("wykres Poincare "+ plik)
#     plt.savefig(os.path.join(KATALOG_WYKRESOW, 'Wykres_poincare_'+plik+'.jpg'), dpi=300 ) 
#     plt.show()
    
#%%wykres poincare dla grup wiekowych
# pliki_lst=[pliki_20,pliki_30,pliki_40,pliki_50,pliki_60,pliki_70,pliki_80]

# for i in range(0,len(pliki_lst)):
#     dane_dekady = []
#     for plik in pliki_lst[i]:
#         seria = load_serie(skad = czytamy, co= plik )
#         dane_dekady.append(seria) 
#     final_df = pd.concat(dane_dekady,ignore_index=True)
#     wykres_poincare(final_df['Interval'].values)
#     plt.title("wykres Poincare dla {}0 latków".format(i+2))
#     plt.savefig(os.path.join(KATALOG_WYKRESOW,'Wykres_poincare_{}0_latków.jpg'.format(i+2)), dpi=300 ) 
#     plt.show()        


#%%funkcja liczaca roznice tylko jezeli wartosc w Num_contraction jest odpowiednia
def funkcja_roznic(df, nr=0):
    poprzedni_numer = df['Num_contraction'][nr]
    poprzednia_wartosc = df['Interval'][nr]
    roznice = []
    roznice_cykl = []
    for index, row in df.iterrows():
        if (row['Num_contraction'] == (poprzedni_numer + 1)):
            roznice_cykl.append(row['Interval']-poprzednia_wartosc)
        else: 
            roznice.append(roznice_cykl)
            roznice_cykl = []
        poprzedni_numer = row['Num_contraction']
        poprzednia_wartosc = row['Interval']
    roznice.append(roznice_cykl)
    return roznice


#%%Poszukiwanie typowych wzorcow
def wyznacz_a_d_0(roznice):
    #lista list
    a_d_0 = []
    for cykl in roznice:
        a_d_0_cykl = []
        for roznica in cykl: 
            if roznica > 0:
                a_d_0_cykl.append('d')
            elif roznica < 0:
                a_d_0_cykl.append('a')
            else:
                a_d_0_cykl.append('0')
        a_d_0.append(a_d_0_cykl)
    return a_d_0


def prawdopodobienstwa_ciagow_jednoelementowych(a_d_0):
    licznik_a = 0
    licznik_d = 0
    licznik_0 = 0
    for cykl in a_d_0:
        for symbol in cykl:
            if symbol == 'a':
                licznik_a += 1
            elif symbol == 'd':
                licznik_d += 1
            else:
                licznik_0 += 1 
    ilosc = licznik_0 + licznik_a + licznik_d
    return {'P(a)': licznik_a/ilosc, 'P(d)':licznik_d/ilosc, 'P(0)':licznik_0/ilosc} 

def prawdopodobienstwa_ciagow_dwuelementowych(a_d_0):
    liczniki = {'aa':0, 'ad':0, 'a0':0, 'dd':0, 'da':0, 'd0':0, '00':0, '0a':0, '0d':0}
    ilosc = 0
    for a_d_0_cykl in a_d_0:
        if len(a_d_0_cykl) == 0:
            continue
        dwuznak = a_d_0_cykl[0]
        for i in range(1, len(a_d_0_cykl)):
            dwuznak += a_d_0_cykl[i]
            liczniki[dwuznak] += 1
            dwuznak = dwuznak[1:]
            ilosc += 1
    prawdopodobienstwa = {}
    for dwuznak in liczniki.keys():
        klucz = f"P({dwuznak})" 
        prawdopodobienstwa[klucz] = liczniki[dwuznak]/ilosc 
    return prawdopodobienstwa

def prawdopodobienstwa_ciagow_trzyelementowych(a_d_0):
    liczniki = {}
    ilosc = 0 
    for znak1 in ['a', 'd', '0']:
        for znak2 in ['a', 'd', '0']:
            for znak3 in ['a', 'd', '0']:
                ciag = znak1 + znak2 +znak3
                liczniki[ciag] = 0
    for a_d_0_cykl in a_d_0:
        if len(a_d_0_cykl) < 2:
            continue
        ciag = a_d_0_cykl[0] + a_d_0_cykl[1]
        for i in range(2, len(a_d_0_cykl)):
            ciag += a_d_0_cykl[i]
            liczniki[ciag] += 1 
            ciag = ciag[1:]
            ilosc += 1
    prawdopodobienstwa = {}
    for ciag in liczniki.keys():
        key = f"P({ciag})"
        prawdopodobienstwa[key] = liczniki[ciag]/ilosc 
    return prawdopodobienstwa
    
#%%analiza sygnalu oraz zroznicowanego sygnalu w przesuwajacych sie oknach

def okna_sygnal(seria, nazwa_pliku):
    odchylenia = []
    odchylenie_max = 0
    nr_okna_od = 0 
    srednie = []
    srednia_max = 0
    nr_okna_sr = 0
    
    interwaly = seria['Interval']
    for i in range(0,len(seria),100):
        if (i + 100) > len(seria):
            break
        okno = interwaly[i:i+100]
        srednia = np.mean(okno)
        srednie.append(srednia)
        odchylenie = np.std(okno)
        odchylenia.append(odchylenie)
        
        if odchylenie > odchylenie_max:
            odchylenie_max = odchylenie
            nr_okna_od =  i
            
        if srednia > srednia_max:
            srednia_max = srednia
            nr_okna_sr = i

    #plt.figure(figsize=(10, 6))
    #plt.title('Analiza sygnalu w oknach ' + nazwa_pliku)
    #sns.scatterplot(x = range(len(srednie)), y = srednie, color = "#7D9EC0")
    #plt.xlabel('odchylenia')
    #plt.ylabel('srednie')
    #plt.savefig(os.path.join(KATALOG_WYKRESOW, 'analiza_sygnalu_'+ nazwa_pliku +'.jpg'), dpi=300 ) 
    
    #okno o największym odchyleniu RR
    results1_od = {}
    
    okno_od = seria.iloc[nr_okna_od:nr_okna_od+100, :].copy()
    
    lista_list_roznic_od = funkcja_roznic(okno_od, nr_okna_od)
    roznice_lista_od = []
    
    interwaly = seria['Interval']
    okno_interwaly_od = interwaly[nr_okna_od:nr_okna_od+100]
    
    for lista_roznic_od in lista_list_roznic_od:
        roznice_lista_od += lista_roznic_od
    roznice_np_od = np.array(roznice_lista_od)
    
    results1_od['min_RR_okno'] = np.min(okno_interwaly_od)
    results1_od['max_RR_okno'] = np.max(okno_interwaly_od)
    results1_od['srednie_RR_okno'] = round(np.mean(okno_interwaly_od),3)
    results1_od['SDNN_okno'] = round(np.std(okno_interwaly_od),3)
    results1_od['RMSSD_okno'] = round(np.sqrt(np.mean(np.square(roznice_np_od))),3)
    results1_od['pNN20_okno'] = round(100*(np.sum(np.abs(roznice_np_od) > 20))/(len(roznice_np_od)),3)
    results1_od['pNN50_okno'] = round(100*(np.sum(np.abs(roznice_np_od) > 50))/(len(roznice_np_od)),3)
    
    a_d_0_od = wyznacz_a_d_0(lista_list_roznic_od)
    
    results2_od = prawdopodobienstwa_ciagow_jednoelementowych(a_d_0_od)
    results3_od = prawdopodobienstwa_ciagow_dwuelementowych(a_d_0_od)
    results4_od = prawdopodobienstwa_ciagow_trzyelementowych(a_d_0_od)
    
    results_od = {**results1_od,**results2_od,**results3_od,**results4_od}
    results_od = {k + '_od': v for k,v in results_od.items()} 
    
    #okno o największym srednim RR
    results1_sr = {}
    
    okno_sr = seria.iloc[nr_okna_sr:nr_okna_sr+100, :].copy()
    
    lista_list_roznic_sr = funkcja_roznic(okno_sr, nr_okna_sr)
    roznice_lista_sr = []
    
    okno_interwaly_sr = interwaly[nr_okna_sr:nr_okna_sr+100]
    
    for lista_roznic_sr in lista_list_roznic_sr:
        roznice_lista_sr += lista_roznic_sr
    roznice_np_sr = np.array(roznice_lista_sr)
    
    results1_sr['min_RR_okno'] = np.min(okno_interwaly_sr)
    results1_sr['max_RR_okno'] = np.max(okno_interwaly_sr)
    results1_sr['srednie_RR_okno'] = round(np.mean(okno_interwaly_sr),3)
    results1_sr['SDNN_okno'] = round(np.std(okno_interwaly_sr),3)
    results1_sr['RMSSD_okno'] = round(np.sqrt(np.mean(np.square(roznice_np_sr))),3)
    results1_sr['pNN20_okno'] = round(100*(np.sum(np.abs(roznice_np_sr) > 20))/(len(roznice_np_sr)),3)
    results1_sr['pNN50_okno'] = round(100*(np.sum(np.abs(roznice_np_sr) > 50))/(len(roznice_np_sr)),3)
    
    a_d_0_sr = wyznacz_a_d_0(lista_list_roznic_sr)
    
    results2_sr = prawdopodobienstwa_ciagow_jednoelementowych(a_d_0_sr)
    results3_sr = prawdopodobienstwa_ciagow_dwuelementowych(a_d_0_sr)
    results4_sr = prawdopodobienstwa_ciagow_trzyelementowych(a_d_0_sr)
    
    results_sr = {**results1_sr,**results2_sr,**results3_sr,**results4_sr}
    results_sr = {k + '_sr': v for k,v in results_sr.items()}  
    
    return results_od, results_sr
    

def okna_roznice(roznice_np, nazwa_pliku):
    srednie = []
    odchylenia = []
    for i in range(0,len(roznice_np),100):
        if (i + 100) > len(roznice_np):
            break
        okno = roznice_np[i:i+100]
        srednia = np.mean(okno)
        srednie.append(srednia)
        odchylenie = np.std(okno)
        odchylenia.append(odchylenie)
        
    #plt.figure(figsize=(10, 6))
    #plt.title('Analiza zroznicowanego sygnalu w oknach '+ nazwa_pliku)
    #sns.scatterplot(x = odchylenia , y = srednie, color = "#7D9EC0")
    #plt.xlabel('odchylenia')
    #plt.ylabel('srednie')
    #plt.savefig(os.path.join(KATALOG_WYKRESOW, 'analiza_zroznicowanego_sygnalu_'+ nazwa_pliku +'.jpg'), dpi=300 )


#%%funkcje do wzorców k
def k_1elementowe(lista_k):
    k_1el = {}
    for k in lista_k:
        if f"{k}" not in k_1el.keys():
            k_1el[f"{k}"] = 1
        else:
            k_1el[f"{k}"] += 1
    return k_1el

    
def k_2elementowe(lista_k):
    k_2el = {}
    for i in range(0,len(lista_k)-1):
        k1 = lista_k[i]
        k2 = lista_k[i+1]
        if f"({k1,k2})" not in k_2el.keys():
            k_2el[f"({k1,k2})"] = 1
        else:
            k_2el[f"({k1,k2})"] += 1
    return k_2el

def k_3elementowe(lista_k):
    k_3el = {}             
    for i in range(0,len(lista_k)-2):
        k1 = lista_k[i]
        k2 = lista_k[i+1]
        k3 = lista_k[i+2]
        if f"({k1,k2,k3})" not in k_3el.keys():
            k_3el[f"({k1,k2,k3})"] = 1
        else:
            k_3el[f"({k1,k2,k3})"] += 1
    return k_3el

#%%
def wlasnosci_sygnalu(df, nazwa_pliku):
    results1 = {}
    
    lista_list_roznic = funkcja_roznic(df)
    roznice_lista = []
    
    for lista_roznic in lista_list_roznic:
        roznice_lista += lista_roznic
    roznice_np = np.array(roznice_lista)
    
    results1['min_RR'] = np.min(df['Interval'])
    results1['max_RR'] = np.max(df['Interval'])
    results1['srednie_RR'] = round(np.mean(df['Interval']),3)
    results1['SDNN'] = round(np.std(df['Interval']),3)
    results1['RMSSD'] = round(np.sqrt(np.mean(np.square(roznice_np))),3)
    results1['pNN20'] = round(100*(np.sum(np.abs(roznice_np) > 20))/(len(roznice_np)),3)
    results1['pNN50'] = round(100*(np.sum(np.abs(roznice_np) > 50))/(len(roznice_np)),3)
    
    a_d_0 = wyznacz_a_d_0(lista_list_roznic)
    
    results2 = prawdopodobienstwa_ciagow_jednoelementowych(a_d_0)
    results3 = prawdopodobienstwa_ciagow_dwuelementowych(a_d_0)
    results4 = prawdopodobienstwa_ciagow_trzyelementowych(a_d_0)
    results = {**results1,**results2,**results3,**results4}
    
    lista_k = [r / 8 for r in roznice_lista]
    slownik_k1 = k_1elementowe(lista_k)
    slownik_k2 = k_2elementowe(lista_k)
    slownik_k3 = k_3elementowe(lista_k)
    
    slownik_k = {**slownik_k1,**slownik_k2,**slownik_k3}
    
    #okna_roznice(roznice_np, nazwa_pliku)
    
    return results, slownik_k

                
#%%
dekady_pom = []
plec_pom = []

for item in pliki:
    dekady_pom.append(item[1:3])
    plec_pom.append(item[0])
    
dekady_int = [int(x) for x in dekady_pom] 

#%%tworzenie głównej tabeli,slownikow dla k,
#ta komórka wywołuje się najdluzej poniewaz korzysta z wiekszosci wczesniejszych funkcji

pliki_lst=[pliki_20_f,pliki_30_f,pliki_40_f,pliki_50_f,pliki_60_f,pliki_70_f,pliki_80_f,
           pliki_20_m,pliki_30_m,pliki_40_m,pliki_50_m,pliki_60_m,pliki_70_m,pliki_80_m]

glowna_tabela =  pd.DataFrame() 
tabela_std = pd.DataFrame() 
tabela_sr = pd.DataFrame()
#lista_slownikow_k = []

for pliki_dekady in pliki_lst: 
    for plik in pliki_dekady:
        seria = load_serie(skad = czytamy, co= plik )
        results = wlasnosci_sygnalu(seria, plik)
        results_okna = okna_sygnal(seria, plik)
        glowna_tabela = glowna_tabela.append(results[0], ignore_index = True)
        tabela_std = tabela_std.append(results_okna[0], ignore_index = True)
        tabela_sr = tabela_sr.append(results_okna[1], ignore_index = True)
        #lista_slownikow_k.append(results[1])
    
glowna_tabela.insert(loc = 0, column = 'nazwa_pliku', value = pliki)
glowna_tabela.insert(loc = 1, column = 'dekada', value = dekady_int)
glowna_tabela.insert(loc = 2, column = 'plec', value = plec_pom)


tabela_std.insert(loc = 0, column = 'nazwa_pliku', value = pliki)
tabela_std.insert(loc = 1, column = 'dekada', value = dekady_int)
tabela_std.insert(loc = 2, column = 'plec', value = plec_pom)

tabela_sr.insert(loc = 0, column = 'nazwa_pliku', value = pliki)
tabela_sr.insert(loc = 1, column = 'dekada', value = dekady_int)
tabela_sr.insert(loc = 2, column = 'plec', value = plec_pom)


#%%zapisanie tabel do pliku
glowna_tabela.to_excel("tabela_bazowa.xlsx",index=False)           
tabela_std.to_excel("tabela_std.xlsx",index=False) 
tabela_sr.to_excel("tabela_sr.xlsx",index=False) 

         
#%%   
# pliki_lst=[pliki_20,pliki_30,pliki_40,pliki_50,pliki_60,pliki_70,pliki_80]

# tabele_dekady = []

# for pliki_dekady in pliki_lst:
#     tabela_dekady =  pd.DataFrame() 
#     for plik in pliki_dekady:
#         seria = load_serie(skad = czytamy, co= plik )
#         results = wlasnosci_sygnalu(seria, plik)
#         tabela_dekady = tabela_dekady.append(results[0], ignore_index = True)
#     tabela_dekady.insert(loc = 0, column = 'nazwa_pliku', value = pliki_dekady)    
#     tabele_dekady.append(tabela_dekady)   
    
#%%Charakterystyki głownej tabeli
# #RR
# print('\nMaksymalne RR ze wszystkich plików: ')
# print(glowna_tabela['nazwa_pliku'][glowna_tabela.max_RR == glowna_tabela.max_RR.max()])
# print(glowna_tabela.max_RR.max())

# print('\nMinimalne RR ze wszystkich plików: ')
# print(glowna_tabela['nazwa_pliku'][glowna_tabela.min_RR == glowna_tabela.min_RR.min()])
# print(glowna_tabela.min_RR.min())

# #srednie RR
# print('\nMaksymalna srednia RR: ')
# print(glowna_tabela['nazwa_pliku'][glowna_tabela.srednie_RR == glowna_tabela.srednie_RR.max()])
# print(glowna_tabela.srednie_RR.max())

# print('\nMinimalna srednia RR: ')
# print(glowna_tabela['nazwa_pliku'][glowna_tabela.srednie_RR == glowna_tabela.srednie_RR.min()])
# print(glowna_tabela.srednie_RR.min())

# #SDNN
# print('\nMaksymalne SDNN: ')
# print(glowna_tabela['nazwa_pliku'][glowna_tabela.SDNN == glowna_tabela.SDNN.max()])
# print(glowna_tabela.SDNN.max())

# print('\nMinimalne SDNN: ')
# print(glowna_tabela['nazwa_pliku'][glowna_tabela.SDNN == glowna_tabela.SDNN.min()])
# print(glowna_tabela.SDNN.min())

# #RMSSD
# print('\nMaksymalne RMSSD: ')
# print(glowna_tabela['nazwa_pliku'][glowna_tabela.RMSSD == glowna_tabela.RMSSD.max()])
# print(glowna_tabela.RMSSD.max())

# print('\nMinimalne RMSSD: ' )
# print(glowna_tabela['nazwa_pliku'][glowna_tabela.RMSSD == glowna_tabela.RMSSD.min()])
# print(glowna_tabela.RMSSD.min())

# #pNN20
# print('\nMaksymalne pNN20: ')
# print(glowna_tabela['nazwa_pliku'][glowna_tabela.pNN20 == glowna_tabela.pNN20.max()])
# print(glowna_tabela.pNN20.max())

# print('\nMinimalne pNN20: ')
# print(glowna_tabela['nazwa_pliku'][glowna_tabela.pNN20 == glowna_tabela.pNN20.min()])
# print(glowna_tabela.pNN20.min())

# #pNN50
# print('\nMaksymalne pNN50: ')
# print(glowna_tabela['nazwa_pliku'][glowna_tabela.pNN50 == glowna_tabela.pNN50.max()])
# print(glowna_tabela.pNN50.max())

# print('\nMinimalne pNN50: ')  
# print(glowna_tabela['nazwa_pliku'][glowna_tabela.pNN50 == glowna_tabela.pNN50.min()])
# print(glowna_tabela.pNN50.min())     
   
#%%tabela do analizy danych z poszczególnych dekad
# analiza_dekad =  pd.DataFrame()

# nazwy = ['min_RR','max_RR',
#          'min_SDNN','max_SDNN',
#          'RMSSD_min','RMSSD_max',
#          'pNN20_min','pNN20_max',
#          'pNN50_min','pNN50_max']        
    
# for i in range(0,len(tabele_dekady)):
#     wart_dekada = []

#     wart_dekada.append(tabele_dekady[i].min_RR.min())
#     wart_dekada.append(tabele_dekady[i].max_RR.max())

#     wart_dekada.append(tabele_dekady[i].SDNN.min())
#     wart_dekada.append(tabele_dekady[i].SDNN.max())
 
#     wart_dekada.append(tabele_dekady[i].RMSSD.min())
#     wart_dekada.append(tabele_dekady[i].RMSSD.max())

#     wart_dekada.append(tabele_dekady[i].pNN20.min())
#     wart_dekada.append(tabele_dekady[i].pNN20.max())

#     wart_dekada.append(tabele_dekady[i].pNN50.min())
#     wart_dekada.append(tabele_dekady[i].pNN50.max())
    
#     wart_dekada = wart_dekada + tabele_dekady[i].loc[:,'RMSSD':'pNN50'].mean().tolist()
#     nazwy2 = ['sr_' + x for x in tabele_dekady[i].loc[:,'RMSSD':'pNN50'].columns.values.tolist()]
    
#     wart_dekada = wart_dekada + tabele_dekady[i].loc[:,'P(0)':'P(ddd)'].mean().tolist()
#     nazwy3 = ['sr_' + x for x in tabele_dekady[i].loc[:,'P(0)':'P(ddd)'].columns.values.tolist()]
    
#     analiza_dekad[f"{i+2}0"] = wart_dekada

# analiza_dekad.insert(loc = 0,column = 'statystyki',value = nazwy + nazwy2 + nazwy3)     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    