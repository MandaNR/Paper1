# Explication détaillée du papier « Bayesian Deep Knowledge Tracing »

Ce document accompagne `root.tex` et résume chaque section, en clarifiant les équations principales.

---

## 1. Abstract
Le résumé présente BDKT : un LSTM bayésien bi-couche qui conserve une distribution de maîtrise par compétence et mesure deux formes d’incertitude : épistémique (modèle) et aléatoire (bruit). Sur un corpus synthétique de 500 k interactions, BDKT augmente l’AUC de 0,83 → 0,87 et réduit l’erreur de calibration (ECE) de moitié.

## 2. Introduction
Rappelle la transition BKT → DKT et le problème : manque de confiance calibrée. BDKT mélange probabilités et réseaux profonds pour résoudre cela.

## 3. Related Work
1. **IRT** : modélise une aptitude statique ($P(\text{correct}) = 1/(1+e^{-(\theta-b)})$).
2. **BKT** : état caché binaire, quatre paramètres : $P(L_0)$, $P(T)$, $P(S)$, $P(G)$.
3. **DKT** : LSTM qui met à jour un état latent $H_n = f(X_n,H_{n-1})$.
4. Extensions : SAKT (attention), DKVMN (mémoire clé-valeur), GKT (graphes de compétences).

## 4. Bayesian Deep Knowledge Tracing (BDKT)
### 4.1 État cognitif bayésien
- Vecteur continu $\mathbf{K}_t \in [0,1]^n$.
- Distribution : $\mathcal{N}(\boldsymbol{\mu}_t,\boldsymbol{\Sigma}_t)$ (Eq. 1).
- Covariance $\Sigma$ guidée par le graphe de prérequis (Eq. 2).

### 4.2 Transition dynamique avec oubli
- Équation (Eq. 3) combine persistance, apprentissage via $f$ (LSTM bayésien) et bruit $\varepsilon_t$.
- Taux d’apprentissage-oubli $\boldsymbol{\lambda}_t$ (Eq. 4) dépend du temps inter-pratique.

### 4.3 Observation multimodale
Probabilité jointe factorisée sur M modalités (Eq. 5).

### 4.4 Inférence variationnelle
ELBO modifiée (Eq. 6) avec pénalités pédagogiques $\mathcal{R}_{\text{mono}}$ et $\mathcal{R}_{\text{transfer}}$.

### 4.5 Attention métacognitive
Deux niveaux d’attention : concepts puis sources (Eq. 7-8) pour agréger l’information.

### 4.6 Cold-Start & Interprétabilité
- Initialisation méta-apprise (Eq. 9).
- Score d’importance (Eq. 10) et confiance (Eq. 11).

### Schéma de l’architecture BDKT (Fig. 1)
Le diagramme minimaliste illustre le flux d’information à chaque interaction :

- **$x_i$ (cercle gris clair)** : entrée encodée (item + réponse) au temps $i$.
- **Cluster(Stu\_Seg$_i$) (boîte pointillée)** : module bayésien qui met à jour la distribution de maîtrise du pas courant en s’appuyant sur BKT.
- **$h_i$ (cercle bleu)** : état caché du LSTM bayésien qui transporte l’historique de la séquence.
- **Flèches horizontales $h_{i}\rightarrow h_{i+1}$** : propagation séquentielle typique d’un RNN.
- **$y_i$ (cercle rouge)** : probabilité prédite de réussite pour le prochain exercice.
- **Boîte « assess ability »** : composant facultatif d’évaluation externe qui peut renseigner la difficulté ou le statut de l’étudiant.

Chaque colonne représente un pas de temps : l’entrée $x_i$ déclenche une mise à jour bayésienne (Cluster) puis l’état $h_i$ est produit et sert à prédire $y_i$. L’ensemble capture à la fois l’incertitude immédiate (niveau compétence) et la dynamique temporelle (LSTM).

## 5. Dataset
Corpus synthétique : 4 000 étudiants, 6 000 items, 30 compétences, 500 952 interactions.

## 6. Experimental Setup
- Architecture : LSTM bayésien (128) + couche probabiliste.
- Entraînement : Adam, ELBO, Monte-Carlo dropout.
- Baselines : BKT, DKT, IRT.

## 7. Results
Tableau 1 compare les métriques ; BDKT obtient AUC 0,87 et ECE 0,04.

## 8. Conclusions
BDKT apporte une incertitude quantifiée au traçage de connaissances et améliore fiabilité et précision.

---

## Notations clés
| Symbole | Signification |
|---------|---------------|
| $\mathbf{K}_t$ | Vecteur de maîtrise des compétences au temps $t$ |
| $\boldsymbol{\mu}_t$, $\boldsymbol{\Sigma}_t$ | Moyenne & covariance de l’état bayésien |
| $\boldsymbol{\lambda}_t$ | Taux apprentissage/oubli |
| $\varepsilon_t$ | Bruit de processus |
| $o_t$ | Observations multimodales |
| ELBO | Evidence Lower Bound |

---

> Ce fichier est un guide rapide. Pour plus de détails, se référer aux équations et descriptions complètes dans `root.tex`. 
