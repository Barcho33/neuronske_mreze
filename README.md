# FER+ Emotion Recognition using ResNet50

## Opis projekta

Ovaj projekat implementira sistem za prepoznavanje emocija sa fotografija lica korišćenjem **pretrained ResNet50** modela i **PyTorch** biblioteke.

Korišćen je **FER+** dataset koji je prethodno:

* očišćen od loše klasifikovanih slika,
* uklonjene su profilne fotografije,
* uklonjene su slike lošeg kvaliteta,
* uklonjena su prekrivena lica,
* slike su raspoređene u foldere po klasama,
* izvršena je offline augmentacija.

Model koristi **Feature Extraction** pristup, gde je backbone ResNet50 zamrznut, a trenira se samo poslednji klasifikacioni sloj.

---

# Struktura projekta

```text
project/
│
├── data/
│   ├── train/
│   │   ├── anger/
│   │   ├── fear/
│   │   ├── happiness/
│   │   ├── neutral/
│   │   ├── sadness/
│   │   └── surprise/
│   │
│   ├── valid/
│   │   ├── anger/
│   │   ├── fear/
│   │   ├── happiness/
│   │   ├── neutral/
│   │   ├── sadness/
│   │   └── surprise/
│   │
│   ├── test/
│   │   ├── anger/
│   │   ├── fear/
│   │   ├── happiness/
│   │   ├── neutral/
│   │   ├── sadness/
│   │   └── surprise/
│   │
│   └── labels.csv
│
├── models/
│
├── src/
│   ├── main.py
│   ├── config.py
│   ├── transforms.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset

Koristi se 6 emocija:

* anger
* fear
* happiness
* neutral
* sadness
* surprise

Originalne slike su:

* grayscale
* 48x48

Tokom učitavanja automatski se:

* konvertuju u RGB,
* resize-uju na 224x224,
* normalizuju prema ImageNet statistici.

Offline augmentacija je već izvršena, tako da se tokom treninga ne koristi dodatna augmentacija.

---

# Moduli

## config.py

Centralizovana konfiguracija projekta.

Sadrži:

* putanje do dataseta,
* parametre treninga,
* parametre modela,
* putanju za čuvanje modela.

---

## transforms.py

Definiše preprocessing slika.

Izvozi:

* train_transform
* val_transform

---

## dataset.py

Odgovoran za učitavanje dataseta.

Izvozi:

* get_train_dataset()
* get_validation_dataset()
* get_test_dataset()
* get_dataloaders()
* get_class_names()
* get_class_to_idx()

Za učitavanje koristi `torchvision.datasets.ImageFolder`.

---

## model.py

Definiše model.

Koristi pretrained ResNet50.

Backbone je zamrznut, a poslednji Linear sloj zamenjen klasifikatorom za 6 emocija.

---

## train.py

Odgovoran za trening modela.

Radi:

* učitavanje DataLoader-a,
* trening,
* validaciju,
* čuvanje najboljeg modela.

Najbolji model se čuva u:

```
models/best_model.pth
```

---

## evaluate.py

Evaluacija modela.

Računa:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix
* F1-score po klasama

---

## main.py

Ulazna tačka projekta.

Podržane komande:

```
python main.py check-data
python main.py train
python main.py evaluate
python main.py train-evaluate
```

---

# Instalacija

## 1. Kloniranje repozitorijuma

```bash
git clone <repo-url>
cd <repo>
```

---

## 2. Kreiranje virtualnog okruženja

Linux/macOS:

```bash
python3 -m venv venv
```

Windows:

```bash
python -m venv venv
```

---

## 3. Aktivacija virtualnog okruženja

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```cmd
venv\Scripts\activate
```

---

## 4. Instalacija biblioteka

```bash
pip install -r requirements.txt
```

---

# Pokretanje projekta

Preći u src direktorijum:

```bash
cd src
```

---

## Provera dataseta

```bash
python main.py check-data
```

---

## Trening

```bash
python main.py train
```

Tokom treninga prikazuju se:

* progress bar,
* loss,
* accuracy,
* validation rezultati.

Najbolji model automatski se čuva u:

```
models/best_model.pth
```

---

## Evaluacija

```bash
python main.py evaluate
```

Biće prikazani:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report

i generisani grafici:

* Confusion Matrix
* F1-score po klasama

---

## Trening + Evaluacija

```bash
python main.py train-evaluate
```

---

# Promena konfiguracije

Sve konfiguracione vrednosti nalaze se u:

```
src/config.py
```

Najvažnije:

```python
NUM_CLASSES
IMAGE_SIZE
BATCH_SIZE
NUM_WORKERS

LEARNING_RATE
EPOCHS

DEVICE

PRETRAINED
FREEZE_BACKBONE
```

---

# Napomene

* Dataset nije deo repozitorijuma.
* Folder `data/` mora biti postavljen ručno prema prikazanoj strukturi.
* Za treniranje se preporučuje NVIDIA GPU sa CUDA podrškom.
* Na CPU-u će trening biti značajno sporiji.
* Najbolji model se automatski čuva i koristi za evaluaciju.
