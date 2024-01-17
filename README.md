Ce dossier est associé au projet qui se trouve à l'adresse suivante: https://github.com/JulienNicola/Projet7

# Utilisation des différents fichiers dans le process de déploiement:	

**github/workflows/main_placedemarche.yml**: coordination des différentes actions de déploiement dans Github et Azure (généré automatiquement)

**data_sample.csv**: échantillon de 10 000 individus inclus dans le jeu d'entrainement de départ pour intégration dans le dashboard

**exp.zip**: Valeurs de Shap explanation pour explication de l'importance des features pour chacun de nos clients (locale)

**logo.png / sum_plot.png**: images pour intégration dans le dashboard: logo de la société et capture d'image de l'importance globale des features selon SHAP

**requirements.txt**: librairies python nécessaires pour l'application

**run.sh**: commande de démarrage fournie à Azure pour l'application web -> permet de définir le port utilisé par Streamlit (443) et d'installer les dépendances nécessaires au bon fonctionnement du modèle LGBM (libgomp1)

**streamplit_app.py**: développement de l'application de dashboarding en utilisant Streamlit

**threshold.pkl**: seuil de probavilité déterminé via notre travail de modélisation afin d'optimiser notre fonction cout métier
