# Symulator Problemu Wielorękiego Bandyty (Multi-Armed Bandit)

Projekt stanowi implementację i środowisko testowe dla klasycznego problemu **Wielorękiego Bandyty (Multi-Armed Bandit)** w uczeniu ze wzmocnieniem (Reinforcement Learning). Aplikacja umożliwia symulację oraz porównanie efektywności trzech popularnych algorytmów decyzyjnych w stacjonarnym środowisku o rozkładzie normalnym (Gaussa).

Głównym celem programu jest optymalizacja wyboru ramion bandyty w celu maksymalizacji skumulowanej nagrody oraz minimalizacji tzw. żalu (**Regret**), czyli straty wynikającej z niewybrania optymalnego ramienia w każdym kroku.

---

## 🧠 Zaimplementowane Algorytmy

Projekt implementuje trzy fundamentalne podejścia do balansu pomiędzy eksploracją (Exploration) a eksploatacją (Exploitation):

1. **Epsilon-Greedy (ε-Greedy)**:
   - Wybiera najlepsze znane ramię z prawdopodobieństwem `1 - ε`.
   - Z prawdopodobieństwem `ε` dokonuje losowej eksploracji środowiska.
   - Wartość `ε` jest konfigurowalna z poziomu plików ustawień.

2. **UCB (Upper Confidence Bound)**:
   - Realizuje zasadę *„optymizmu w obliczu niepewności”*.
   - Wybiera ramię na podstawie średniej dotychczasowej nagrody powiększonej o przedział ufności.
   - Automatycznie zmniejsza priorytet eksploracji dla ramion, które były już wielokrotnie sprawdzane.

3. **Próbkowanie Thompsona (Thompson Sampling)**:
   - Podejście bayesowskie wykorzystujące próbkowanie z rozkładu a posteriori.
   - Ze względu na ciągły, gaussowski charakter nagród, algorytm wykorzystuje koniugację rozkładów **Normal-Inverse-Gamma (NIG)**.
   - Dynamicznie aktualizuje hiperparametry (`μ`, `ν`, `α`, `β`) na podstawie obserwowanych nagród.

---

## 📁 Struktura Projektu

Projekt został zaprojektowany w sposób modułowy, z wyraźnym odseparowaniem logiki środowiska od algorytmów:

```text
├── bandit/
│   └── bandit.py          # Definicja środowiska (ramiona, średnie, odchylenia standardowe, losowanie nagród)
├── algorithms/
│   ├── algorithm.py       # Abstrakcyjna klasa bazowa dla algorytmów
│   ├── e_greedy.py        # Implementacja algorytmu Epsilon-Greedy
│   ├── ucb.py             # Implementacja algorytmu UCB
│   └── thompson.py        # Implementacja Próbkowania Thompsona z aktualizacją NIG
├── data/
│   ├── 01.json            # Konfiguracja środowiska testowego nr 1 (4 ramiona)
│   └── 02.json            # Konfiguracja środowiska testowego nr 2 (9 ramion)
├── settings.json          # Globalne ustawienia symulacji (liczba gier, wartość epsilon)
└── main.py                # Główny skrypt uruchomieniowy i wizualizacja wyników
```

---

## 🛠️ Wymagania i Instalacja

Program wymaga środowiska Python 3.8 lub nowszego oraz zainstalowanych bibliotek numerycznych i graficznych.

1. Sklonuj repozytorium lub pobierz pliki projektu.
2. Zainstaluj wymagane pakiety za pomocą menedżera `pip`:

```bash
pip install numpy matplotlib
```

---

## ⚙️ Konfiguracja Symulacji

### Ustawienia ogólne (`settings.json`)

W tym pliku definiowane są globalne parametry uruchomienia:

- `epsilon`: współczynnik eksploracji dla algorytmu `ε-Greedy`.
- `games`: łączna liczba rund (iteracji/pociągnięć za ramię), z których składa się jedna symulacja.

```json
{
    "epsilon": 0.1,
    "games": 500
}
```

### Definicje Środowiska (`data/02.json`)

Pliki JSON w folderze `data/` definiują strukturę nagród dla wielorękiego bandyty:

- `arms`: liczba dostępnych ramion.
- `mean`: oczekiwana średnia wartość nagrody dla każdego ramienia.
- `std`: odchylenie standardowe nagrody (szum gaussowski) dla każdego ramienia.

---

## 🚀 Uruchomienie Programu

Aby uruchomić symulację i porównać algorytmy, wykonaj polecenie:

```bash
python main.py -f 01.json
```

### Oczekiwany Wynik w Konsoli

Po zakończeniu symulacji program wypisze w terminalu podsumowanie zdobytych nagród oraz wynik teoretycznego „jasnowidza” (wybierającego wyłącznie najlepsze ramię przez całą grę):

```text
Nagrody zdobyte przez algorytm epsilon Greedy: 524.321948210342
Nagrody zdobyte przez algorytm UCB: 612.871239847102
Nagrody zdobyte przez próbkowanie Thompsona: 634.192048123951
Nagroda przy jasnowidzeniu: 650.0
```

### Wizualizacja Wyników

Program automatycznie wygeneruje i wyświetli wykres przedstawiający **skumulowany żal (Total Regret)** dla każdego z algorytmów w czasie.

- *Im bardziej płaski wykres (mniejsze nachylenie w późniejszych fazach), tym szybciej algorytm zidentyfikował optymalne ramię i przestał marnować próby na gorsze opcje.*
