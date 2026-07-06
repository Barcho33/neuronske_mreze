# Podešavanje razvojnog okruženja

## 1. Kloniranje repozitorijuma

``` bash
git clone <URL_REPOZITORIJUMA>
cd <NAZIV_PROJEKTA>
```

------------------------------------------------------------------------

## 2. Kreiranje virtualnog okruženja

Na Linux-u i macOS-u:

``` bash
python3 -m venv venv
```

Na Windows-u:

``` powershell
python -m venv venv
```

------------------------------------------------------------------------

## 3. Aktivacija virtualnog okruženja

### Linux / macOS

``` bash
source venv/bin/activate
```

### Windows (PowerShell)

``` powershell
venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)

``` cmd
venv\Scripts\activate.bat
```

------------------------------------------------------------------------

## 4. Instalacija zavisnosti

Sve potrebne biblioteke nalaze se u datoteci `requirements.txt`.

Instalacija:

``` bash
pip3 install -r requirements.txt
```

------------------------------------------------------------------------

## 5. Provera instalacije

Možete proveriti da li su biblioteke uspešno instalirane:

``` bash
pip3 list
```

ili

``` bash

pip3 freeze
```

------------------------------------------------------------------------

## 6. Deaktivacija virtualnog okruženja

Kada završite sa radom:

``` bash
deactivate
```



