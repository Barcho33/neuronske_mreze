"""
evaluate.py
Odgovoran za evaluaciju istreniranog modela na test skupu.
Koristi: get_dataloaders(), get_model(), get_class_names()
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config import (
    DEVICE,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    BATCH_SIZE,
)
from dataset import get_dataloaders, get_class_names
from model import get_resnet50


#ucitavanje modela
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

def load_model(path: str) -> torch.nn.Module:
    # kreiramo praznu resnet50 arhitekturu bez tezina
    model = get_resnet50(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=False)
    # ucitavamo istreniran model, map location=device automatski konvertuje gpu/cpu

    model.load_state_dict(torch.load(path, map_location=device)) 
    model = model.to(device)
    # menja se u evaluation mod, prilagodjava neke slojeve, recimo Dropout i BatchNorm slojeve iskljucuje
    model.eval() 
    return model


# inferencija na test skupu, odnosno predvidjanje rezultata na nepoznatim podacima

def predict(model, test_loader):
    sve_predikcije = []
    sve_labele = [] # stvarne vrednosti

    with torch.no_grad(): # iskljucujemo racunanje gradijenata, ne pamtimo sta se radi dalje
        for slike, labele in test_loader:# prolazi batch po batch
            slike = slike.to(device)
            izlazi = model(slike) # nizovi sansi po emociji
            # u jednodim nizu nalazimo najvecu sansu i prebacujemo na cpu da bi mogo numpy niz
            pred = izlazi.argmax(dim=1).cpu().numpy()
            sve_predikcije.extend(pred) # dodajeo u listu
            sve_labele.extend(labele.numpy()) # dodajemo

    # vracamo sve stvarne i predvidjene vrednosti
    return np.array(sve_labele), np.array(sve_predikcije) 


# metrike

def racunaj_metrike(y_true, y_pred, class_names):
    #tacnost
    acc = accuracy_score(y_true, y_pred) 
    # preciznost
    # mozda ne al average weighted uyima u obzir broj slika u klasi
    # zero_division=0 error handling da se spreci deljenje sa nulom 
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) # 
    # odziv
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) 
    # f1 score
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

  
    print("EVALUACIJA NA TEST SKUPU")
    print(f"Accuracy:           {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision (weighted){precision:.4f}")
    print(f"Recall    (weighted){recall:.4f}")
    print(f"F1-score  (weighted){f1:.4f}")
    print()
    print("-" * 50)
    print("classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return acc, precision, recall, f1


# confusion matrix

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="matrica_konfuzije.png"):
    cm = confusion_matrix(y_true, y_pred)

    # normalizovana verzija da bi se sve klase tretirale fer, izbalansirali al aj
    # gledamo procenat pogodjenih u okviru klase sad da ne dodje do zabune koja se klasa bolje prepoznaje koja losije
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    # prozor za heatmapove
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # levo apsolutni brojevi
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        # nazivi klasa na osama
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0]
    )
    axes[0].set_title('Matrica konfuzije - apsolutni brojevi')
    axes[0].set_xlabel('predikcija')
    axes[0].set_ylabel('tačna klasa')

    # desni heatmap je normalizovano (0.0 - 1.0)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1]
    )
    axes[1].set_title('Matrica konfuzije - normalizovano')
    axes[1].set_xlabel('predikcija')
    axes[1].set_ylabel('tačna klasa')

    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"matrica konfuzije sacuvana: {save_path}")


#F1 po klasi - bar chart
# odmah se vidi koja klasa najteza a koja najlaganija za nas model u a najlaganije

def plot_f1_po_klasi(y_true, y_pred, class_names, save_path="f1_po_klasi.png"):
    # nije weighted sad negopojedinacni f1 po klasi
    f1_klase = f1_score(y_true, y_pred, average=None, zero_division=0)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_names, f1_klase, color='steelblue')

   
    for bar, val in zip(bars, f1_klase):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.3f}', #zaokruzili na 3 dec
            ha='center', va='bottom', fontsize=10
        )

    plt.ylim(0, 1.1) # jer max f1 score je 1
    plt.title('F1-score po klasi')
    plt.xlabel('Klasa')
    plt.ylabel('F1-score')
    plt.tight_layout()
    # cuva se slika
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"f1 po klasi sačuvan: {save_path}")


def main():
    class_names = get_class_names()
    _, _, test_loader = get_dataloaders() 

    print(f"ucitavanje modela: {MODEL_SAVE_PATH}")
    model = load_model(MODEL_SAVE_PATH)

    print(f"evaluacija na {len(test_loader.dataset)} test slika...\n")
    y_true, y_pred = predict(model, test_loader)

    acc, precision, recall, f1 = racunaj_metrike(y_true, y_pred, class_names)

    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_f1_po_klasi(y_true, y_pred, class_names)


if __name__ == "__main__":
    main()