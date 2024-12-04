import numpy as np
import re

def analiza_semantyczna(dokumenty, zapytanie, k):
    przetworz = lambda tekst: re.sub(r'[^\w\s]', '', tekst.lower())
    terminy = sorted(set(przetworz(termin) for dokument in dokumenty for termin in dokument.split()))
    indeks_terminow = {termin: i for i, termin in enumerate(terminy)}
    macierz_czestotliwosci = np.zeros((len(terminy), len(dokumenty)))
    for j, dokument in enumerate(dokumenty):
        for termin in przetworz(dokument).split():
            if termin in indeks_terminow:
                macierz_czestotliwosci[indeks_terminow[termin], j] = 1
    U, S, Vt = np.linalg.svd(macierz_czestotliwosci, full_matrices=False)
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    V_k = Vt[:k, :]
    dokumenty_zredukowane = np.dot(S_k, V_k)
    wektor_zapytania = np.zeros(len(terminy))
    for termin in przetworz(zapytanie).split():
        if termin in indeks_terminow:
            wektor_zapytania[indeks_terminow[termin]] = 1
    zapytanie_zredukowane = np.dot(np.dot(wektor_zapytania, U_k), np.linalg.inv(S_k))
    podobienstwa = [
        float(
            np.dot(zapytanie_zredukowane, dokumenty_zredukowane[:, i]) / 
            (np.linalg.norm(zapytanie_zredukowane) * np.linalg.norm(dokumenty_zredukowane[:, i]))
        )
        for i in range(dokumenty_zredukowane.shape[1])
    ]
    return [round(podobienstwo, 2) for podobienstwo in podobienstwa]

def main():
    liczba_dokumentow = int(input())
    dokumenty = [input() for _ in range(liczba_dokumentow)]
    zapytanie = input()
    k = int(input())
    podobienstwa = analiza_semantyczna(dokumenty, zapytanie, k)
    print(podobienstwa)

if __name__ == "__main__":
    main()
