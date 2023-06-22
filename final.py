#========================================================================
# Import packages

import streamlit as st
#!pip install PyPDF2
#from PdfReader import PdfFileReader
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, LassoCV,Lasso
from mlxtend.plotting import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from PIL import Image
#%matplotlib inline

from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline

#import matplotlib.pyplot as plt
#import tkinter as tk
#==> Use tex font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

st.set_page_config(page_title= "KMPE_satisfaction", page_icon= ":tada:", layout= "wide" )

# header setting


st.title("SATISFACTION CLIENTS")
st.subheader("Projet de fin de formation en data science")

#================================================================================
#import base64


#st.markdown("![Alt Text](https://media3.giphy.com/media/WlUlR0GdKlr5F3JKBd/giphy.gif?cid=ecf05e4738lqascq7f2w76pp8fquwjtc3ms6p4uybmf43ojr&rid=giphy.gif&ct=g)")


#================================================================================
#import streamlit as st
#from PIL import Image



#================================================================================

#st.write("Mohamed KRIM")
#st.write("Esso-passi PALI")
#st.write("[learn more >] (https://github.com/eppali/DS_satisfaction_KMPE)")

filedirectory = "/Users/KYM/Desktop/st_v5/"
#================================================================================
#st.sidebar.radio("SOMMAIRE", options= [])
#st.sidebar.radio('Sommaire', options=["Introduction", "Contexte et objectifs","Analyse exploratoire des données","Mise en place de modèles","Conclusion"])

#st.sidebar.radio('Sommaire', options=["Introduction"])

#st.sidebar.checkbox('special')
# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#   df = pd.read_csv(uploaded_file)
#   #st.write(dataframe)
#   st.write(df)

def DemoClassif_model(clf,embed_method,df_feature_variable,df_target_variable):
    encode_y=LabelEncoder()
    y_encoded=encode_y.fit_transform(df_target_variable)

    # Create a pipeline
    pipe_text=make_pipeline(embed_method,clf)
    pipe_text.fit(df_feature_variable,y_encoded)

    return encode_y,pipe_text
#================================================================================


#import streamlit as st

def main():

    content_options = ["Thème","Contexte , problématique et enjeux", "Présentation des données et méthodes", 
                       "Analyse exploratoire des données","Modélisation du NPS",
                       "Analyse des verbatims","Classification des commentaires", "Simulateur de la classification",
                       "Regard critique et perspectives","Conclusion","Auteurs du projet"]
    choice = st.sidebar.radio("Sommaire", content_options)


    #================================================================================
    if choice == "Contexte , problématique et enjeux":
        st.header("Contexte , problématique et enjeux")
        st.write("Dans le cadre de notre formation Data scientiste,nous avons entrepris un projet baptisé CsePy Customer Satisfaction Europe, pour évaluer la satisfaction client vis- à-vis d’un organisme financier en Europe. Le projet CsePy a pour objectif d’analyser la base de données fournie par BNP Paribas et réalisé par une société internationale de marketing d’opinion dans 8 pays en Europe (Portugal, L’Espagne, La France, l’Italie, la Belgique, l’Allemagne, la Pologne et le UK). Le but de ce projet est multiple:")
        st.write("1 - Comprendre la note de recommandation attribué par les clients des organismes financiers (banque, crédit spécialiste et les captive auto)")
        st.write("2 - Evaluer l’impact de chaque driver de la recommandation sur la note globale")
        st.write("3 - Analyser les verbatims pour identifier les tendances et les problèmes émergents")
        st.write("4 - Proposer un modèle de machine learning prédictif de la satisfaction des clients en fonction des verbatims analysés")
        st.write("Les résultats attendus de ce projet sont une meilleure compréhension des facteurs qui influencent la satisfaction des clients des organismes financiers en Europe, ainsi que des solutions proposées pour améliorer leur expérience. En outre, le projet permettra de fournir des données exploitables à l’équipe marketing pour leur permettre d’optimiser leur stratégie et d’augmenter leur taux de satisfaction client")
    #================================================================================
    
    
    #================================================================================
    if choice == "Thème":
        st.sidebar.title("Projet CsePy: Satisfaction client")
        #video_file = open('media_ds.mp4', 'rb')
        #video_bytes = video_file.read()
    
        #st.video(video_bytes)
        
        image_welcome = Image.open("welcome.PNG")
        st.image(image_welcome, caption=" ")
    
    
    #================================================================================
    elif choice == "Présentation des données et méthodes":
        
        st.header("La base des données")
        
        st.write()
        st.write("Apperçu des premières lignes du tableau des données")
        df_all_data = pd.read_csv(filepath_or_buffer="statisfaction_alldtata.csv")
        df_all_data = df_all_data.drop(columns=["Unnamed: 0"])
        st.write(df_all_data.head(5))
        
        st.write()
        st.write("Apperçu des dernières lignes du tableau des données")
        st.write()
        st.write(df_all_data.tail(5))
        #
        st.header("Description du Net promotor Score (NPS)")
        st.write("La collection des données se fait exclusivement en ligne via un questionnaire d’une durée de 12 minutes environs, les interviews se font en « Blind » c’est-à-dire de façon anonyme: le répondant ne sait pas la marque qui l’interroge.")
        st.write()
        st.write("Pour garantir que les résultats de l’étude peuvent être généralisés, l’échantillon est représentatif de la population de chaque pays où l’étude est menée. La population cible est comme étant les personnes à partir de 18 ans, qui sont en cours de remboursement d’un crédit ou qui ont fini de rembourser leur crédit dans les 12 derniers mois, le type de crédit est le prêt personnel, crédit renouvelable ou dette consolidée, le crédit hypothécaire est exclu de l’étude. Le crédit peut être direct, par le biais d’un partenaire, chez un concessionnaire, avec/sans carte de paiement.")
        st.write()
        st.write("Pour mesurer la perception des clients, les entreprises posent une question simple : ")
        st.write()
        st.write("**Recommanderiez vous la marque à un membre de votre famille ou à un ami ?**")
        
        st.write("Les répondants donnent une note entre 0 et 10;")
        st.write("0 signifie que vous ne recommanderiez pas du tout cet organisme;")
        st.write("10 signifie que vous recommanderiez tout à fait cet organisme;")
        
        image_prom_VS_det = Image.open("repartitionnote.png")
        st.image(image_prom_VS_det, caption="Catégorie des clients en fonction de leurs notes.")
        
        st.write("**Les Promoteurs**")
        st.write("Répondent avec une note de 9 ou 10, ce sont généralement des clients très satisfaits et qui restent fidèles à la marque")
        
        st.write("**Les Passifs**")
        st.write("Répondent avec une note de 7 ou 8, ils sont moyennement satisfaits du service en général, mais ils n'hésitent pas de changer de marque et aller à la concurence")
        
        st.write("**Les Détracteurs**")
        st.write("Répondent avec une note de 0 à 6, il s'agit surtout des clients mécontents qui sont peu susceptibles de renouvler leur expérience avec la marque, ils peuvent même décourager leur entourage et d'autres clients à faire appel aux service et produit de la marque.")
        
        st.write("La note de recommandation attribuée par un client est un excellent indicateur pour les entreprises mais seule ne suffit pas pour comprendre le ressenti profond de tous les clients; pour avoir donc plus d’explication et aller au-delà des données chiffrées, les entreprises qui mesurent le NPS posent une question ouverte pour permettre aux clients de justifier leur notes de recommandation émises auparavant ou à confier leurs expériences sur un sujet précis. A titre d’exemple, la question suivante: ")
        st.write("**Quelles sont toutes les raisons pour lesquelles vous avez donné cette note ?**")
        

    #================================================================================
    elif choice == "Analyse exploratoire des données":
        st.title("Analyse exploratoire des données")
        
        exploratoire_options = ["Interviews par pays", "Part de marché de souscription par organisme de crédit" , "Répartition des âges des clients par orgasime de crédits", 
                                "Distribution des clients par âge","Notes des clients par pays",
                                "Score NPS par organisme de crédit et par pays","Niveau d'effort des clients par organisme de crédit"]
        #explor_choice = st.selectbox("selectbox",exploratoire_options)

#        explor_choice = st.checkbox("checkbox",exploratoire_options)
        
#----------------------------------------------------------------------------
        st.header("1) Interviews par pays")
              
        country_interview_chbox = st.checkbox("Interviews par pays")
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          
        if country_interview_chbox:
            image_nb_interviews = Image.open("figinterviewsByContry.png")
            st.image(image_nb_interviews, caption="Nombre d'interviews par pays")
            
            #with st.expander("Voir commentaire"):
            st.write("==>  Huit (8) pays ont participé à l’étude:UK, France, Italie, Allemagne, Espagne, Belgique, Pologne et le Portugal;")
            
            st.write()
            st.write("==>  Le niveau de réponse est pratiquement égal dans 5 pays, environs 1800 interviews car des quotas d’interview ont été fixés, moins en Pologne et le Portugal car les populations y sont moins importantes;")

            st.write()
            st.write("==>  Le royaume-Uni possède le niveau d’interviews le plus élevé avec 2196.")

#----------------------------------------------------------------------------    
#----------------------------------------------------------------------------
        st.header("2) Part de marché de souscription par organisme de crédit")
        
        part_marche_souscription_chbox = st.checkbox("Part de marché de souscription par organisme de crédit")
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          
        if part_marche_souscription_chbox:
            image_age_client_VS_competitors = Image.open("figCompetitorType.png")
            st.image(image_age_client_VS_competitors, caption="Répartition de la part de marché de souscription par organisme de crédit")

            #with st.expander("Voir commentaire"):
            st.write("==>  La banque est le canal le plus utilisé pour souscrire à un crédit")


#----------------------------------------------------------------------------    
#----------------------------------------------------------------------------
        st.header("3) Distribution globale des clients par âge")
        
        distribution_client_par_age_chbox = st.checkbox("Distribution globale des clients par âge")
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          
        if distribution_client_par_age_chbox:
            image_nb_clients_VS_age = Image.open("figDistributionAge.png")
            st.image(image_nb_clients_VS_age, caption="Distribution du nombre de clients par rapport à l'âge.")

            #with st.expander("Voir commentaire"):
            st.write("==>  Le portefeuille des organismes de crédit est sur-représenté par la population entre 34 et 60 ans.")

#----------------------------------------------------------------------------    
#----------------------------------------------------------------------------
        st.header("4) Répartition des âges des clients par orgasime de crédits")
        
        repartition_age_client_par_organisme_chbox = st.checkbox("Répartition des âges des clients par orgasime de crédits")
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          
        if repartition_age_client_par_organisme_chbox:
            image_nb_clients_VS_age_VS_competirtor = Image.open("figAgeOrganisme.png")
            st.image(image_nb_clients_VS_age_VS_competirtor, caption="Ages des clients par organisme de crédit")

            #with st.expander("Voir commentaire"):
            st.write("==>  Pas de différence d’âge significative par rapport au canal et la business \n line, une légère différence est observé sur les captive auto où la clientèle s’approche des 50 ans.")

#----------------------------------------------------------------------------    
#----------------------------------------------------------------------------
        st.header("5) Notes des clients par pays")
        
        notes_client_par_pays_chbox = st.checkbox("Notes des clients par pays")
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          
        if notes_client_par_pays_chbox:
            image_NPS_Country = Image.open("figContryA1Distribution.png")
            st.image(image_NPS_Country, caption="Distribution des notes de satisfaction des clients par pays.")

            #with st.expander("Voir commentaire"):
            st.write("==>  Les niveaux de recommandation sont variables selon les pays.")

#----------------------------------------------------------------------------    
#----------------------------------------------------------------------------
        st.header("6) NPS par organisme de crédit et par pays")
        
        nps_par_organisme_credit_par_pays_chbox = st.checkbox("NPS par organisme de crédit et par pays")
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          
        if nps_par_organisme_credit_par_pays_chbox:
            image_score_NPS_Country = Image.open("ScorNPSDataViz.png")
            st.image(image_score_NPS_Country, caption="Score NPS par organisme de crédit et par pays.")

            #with st.expander("Voir commentaire"):
            st.write("==>  La satisfaction des clients est en général plus marquée au niveau des banques.")

#----------------------------------------------------------------------------    
#----------------------------------------------------------------------------
        st.header("7) Niveau d'effort des clients par organisme de crédit")
        
        niveau_effort_client_par_organisme_chbox = st.checkbox("Niveau d'effort des clients par organisme de crédit")
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          
        if niveau_effort_client_par_organisme_chbox:
            image_score_NPS_Country = Image.open("figCustomerEffort.png")
            st.image(image_score_NPS_Country, caption="Distribution du nNiveau d'effort des clients par organisme de crédit.")

            #with st.expander("Voir commentaire"):
            st.write("==>  Les niveaux d’effort des clients sont les moins pour les banques, ce qui peut expliquer en partie cette dominance de satisfaction sur les marchés.")

#----------------------------------------------------------------------------    


    #================================================================================
    elif choice == "Modélisation du NPS":
        st.header("Modélisation du NPS")
        
        
        
        exploratoire_options = ["Préparation de la base", "Récap des modéles" , "Résultats"]
        explor_choice = st.selectbox("selectbox",exploratoire_options)
        
        if explor_choice == "Préparation de la base":
           st.write("**Nettoyage de la base**")
           st.write("La base initiale contenait des données sensibles concernant des informations sur les clients et les compétiteurs")
           st.write(" Nous avons supprimé les informations sensibles des clients, et anonymisé les information sur les compétiteurs")
           st.write("Nous avons fait cela pour les 5 années 2017, 2018, 2019, 2020, 2021")
        
           st.write(" Pour avoir une analyse homogéne entre les pays nous avons supprimé les variables locales qui concernait des attributs spécifique à un pays")
        
           st.write("**La base d'analyse**")
           st.write("Nous avons commencé par consolidé les 5 bases mais aprés reflexion nous avons séparé l'analyse par année")
           st.write("car aprés chaque étude des plans d'action sont mené pour améliorer le service et donc l'étude de l'année suivante sert à mesurer l'éfficacité des plans d'action réalisés")
           st.write(" Nous avons commencé par 2021 la base la plus récente pour répondre qu besoin de l'entreprise cette base contient 13 884 observations et 132  / 288 colonnes ont été gardées ")
        
           st.write("**Gestion des NAN**")
           st.write("les données represente des scores de 1 à 10, comme presenté dans la definition du NPS le score est basé sur la diffèrence entre la part des promoteurs et des detracteurs")
           st.write("il serait dommage de supprimer les NAN, pour garder donc l'exaustivité de la base nous avons remplacé les NAN par le chiffre 7 et non pas le mode ou la moyenne ca ça ne changera rien dans le score NPS")
           st.write(" en plus un client n'est pas obligé de posséder tous les produits ou tester tous les produits de la marque avec qui il est en relation")
           
           st.write("**Les variables**")
        
           image_benchvar = Image.open("benchmark_variables.png")
           st.image(image_benchvar, caption="Benchmark Barometer Engagement Framework")
        
           
           st.write("")
           st.write("-------------------------------------------------")
           
        elif explor_choice == "Récap des modéles":
           st.write("**Comparaison entre les modèles**")
           
           image_recapmodel = Image.open("figcomparmectrics.png")
           st.image(image_recapmodel, caption="Comparaison entre modèle NPS")
           
           st.write("Le Xgboost apparait comme le meilleur modèle en terme de minimisation de l'erreur,")
            
           st.write("-------------------------------------------------")
           
        elif explor_choice == "Résultats":
           st.write("**Variables Principales**")
           
           image_xgboostglobal = Image.open("xgboostglobal.png")
           st.image(image_xgboostglobal, caption="Resultat de regression Xgboost des drivers globales de NPS")
           
           
           st.write("**Variables Detaillées**")
           
           image_xgboostdetail = Image.open("xgboostdetail.png")
           st.image(image_xgboostdetail, caption="Resultat de regression Xgboost des items detaillés de NPS")
           
           image_benchvar = Image.open("benchmark_variables.png")
           st.image(image_benchvar, caption="Benchmark Barometer Engagement Framework")
    #================================================================================
    elif choice == "Analyse des verbatims":
        st.title("Analyse des verbatims")
  
        #°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
        #°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
        st.header("1. Nettoyage des commentaires")
        
        st.write()
        st.write("==> Traduction de tous les commentaires en anglais;")
        
        st.write()
        st.write("==> Suppression des données manquantes (NaNs);")
        
        st.write()
        st.write("==> Suppression des stopwords, des pontuations et certains mots n'ayant pas de sens;")
    
        df_demo_multinominal = pd.read_csv(filepath_or_buffer="base2021multinominal.csv")
        df_demo_multinominal = df_demo_multinominal.drop(columns=["Unnamed: 0", "real_polarity","real_polarity_class","cut_words","letters","comment_length"])
        

        st.write(df_demo_multinominal.head())

        
        
        st.header("2. Nuages de mots ")
    
        columns_wc = ["Tous les commentaires", "Commentaires positifs", "Commentaires négatifs"]
        
        wc_all_chbox = st.checkbox(columns_wc[0])
        wc_pos_chbox = st.checkbox(columns_wc[1])
        wc_neg_chbox = st.checkbox(columns_wc[2])

        col_all, col_pos, col_neg = st.columns(3)
        #col_all, col_pos, col_neg = st.sidebar.beta_columns(3)
        #col_all, col_pos, col_neg = st.beta_columns(3)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with col_all:
            #st.write("Nuage de points global")
            
            if wc_all_chbox:
                #st.image(filedirectory+"/"+"figwc_all.png" , use_column_width=True)
                image_wc_all_cmts = Image.open("figwc_all.png")
                st.image(image_wc_all_cmts,use_column_width=True, caption="Nuage de mots global.")

                st.write()
                st.write("Les clients en global parlent de leur banque, du taux d’intérêt, leur crédit, et des process")

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with col_pos:
            #st.write("Nuage de points pour les commentaires positifs")
                            
            if wc_pos_chbox:
                image_wc_pos_cmts = Image.open("figwcpromotors.png")
                st.image(image_wc_pos_cmts,use_column_width=True, caption="Nuage de mots des commentaires positifs.")
                
                st.write()
                st.write("Les raisons de satisfaction des promoteurs sont liées à la facilité et la rapidité des processus, à un taux d’intérêt attractif.")

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with col_neg:
            #st.write("Nuage de points pour les commentaires négatifs")
                                        
            if wc_neg_chbox:
                image_wc_neg_cmts = Image.open("figwcdetractors.png")
                st.image(image_wc_neg_cmts, use_column_width=True,caption="Nuage de mots des commentaires négatifs.")
                
                st.write()
                st.write("Les détracteurs sont non satisfaits à cause du taux d’intérêt très élevé et une mauvaise expérience avec les services clients de la banque")

        #°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
        #°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
                              
        st.header("3. Représentation par un modèle de N-grams")
        
# =============================================================================
        columns_ngrams = ["Bigrams", "Trigrams"]
        
        bigram_chbox = st.checkbox(columns_ngrams[0])
        trigram_chbox = st.checkbox(columns_ngrams[1])
        

        col_bigram, col_trigram = st.columns(2)

        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with col_bigram:
            #st.write("Nuage de points global")
            
            if bigram_chbox:
                image_bigram = Image.open("figbigram.png")
                st.image(image_bigram,use_column_width=True, caption="Représentation par des bigrams.")

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with col_trigram:
                            
            if trigram_chbox:
                image_trigram = Image.open("figtrigram.png")
                st.image(image_trigram,use_column_width=True, caption="Représentation par des trigrams.")
 
        st.write("Le taux d’intérêt est le sujet le plus utilisé dans les verbatims clients, c’est l’un des éléments qui justifie la recommandation.	")
# =============================================================================
            
    #=====================================================================================++++++++++++++++
    #=====================================================================================++++++++++++++++
    
    elif choice == "Classification des commentaires":
        st.title("Classification des commentaires")

        
        comment_classification_options = ["Classification logistique multinominale", "Classification binaire"]
        #comment_classification_choice = st.selectbox("selectbox",comment_classification_options)
        
        #if comment_classification_choice == "Classification logistique multinominale":
        
        #if comment_classification_choice == "Classification logistique multinominale":
        st.header("A - Classification logistique multinominale")
        
        st.write()
        st.write("==> Classification à trois (3) classes effectuée")
        st.write()
        st.write("==> Classification supervisée effectuée: labels initiaux à partir de la note de satisfaction!")
        st.write()

        st.write()
        st.write("==> Classe négative => label = 0")
        st.write()
        st.write("==> Classe neutre (passifs) => label = 1")
        st.write()
        st.write("==> Classe positive => label = 2")
        df_demo_multinominalclass = pd.read_csv(filepath_or_buffer="base2021classmulti.csv")
        df_demo_multinominalclass = df_demo_multinominalclass.drop(columns=["Unnamed: 0"])
        st.write(df_demo_multinominalclass.head())
        
    
        st.write()
        st.write("==>Trois (3) modèles proposés: Régression logistique, Random Forest et XGBoost")
        st.write()
        st.write("==> Pour chaque modèle, quatre (4) méthodes d'embedding utilisées: Bag Of Words (BOW), Term Frequency - Inverse document Frequency (Tf-Idf), Word2Vec et Doc2Vec")
        
#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

        # Logistic regression
        
        st.subheader("1. Régression logistique (classification logistique multinominale)")
        
        
        columns_tert_class_lreg = ["Matrices de confusion pour la régression logistique ", "Comparaison des méthodes de W-Embedding pour la régression logistique"]
 
        cm_tert_class_lreg_chbox = st.checkbox(columns_tert_class_lreg[0])
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        report_tert_class_lreg_chbox = st.checkbox(columns_tert_class_lreg[1])
        
        col_cm_tert_lreg, col_report_tert_lreg = st.columns(2)
        
        with col_cm_tert_lreg:
            
            if cm_tert_class_lreg_chbox:
                image_CM_teriary_class_logreg = Image.open("figCMLogisticregression.png")
                st.image(image_CM_teriary_class_logreg, caption="Classification logistique multinominale: matrices de confusion pour différentes méthodes de word embedding avec la régression logistique.")
                st.write()
                st.write("Parmi les bonnes prédictions, seules la classe 'neutre' atteint les 50% à l'exception de la prédiction de la classe négative avec régression logistique.")
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with col_report_tert_lreg:
                
            if report_tert_class_lreg_chbox:
                image_Report_teriary_class_logreg = Image.open("figReportLogisticregression.png")
                st.image(image_Report_teriary_class_logreg, caption="Classification logistique multinominale: comparaison entre les méthodes de word embedding avec la régression logistique.")
                st.write()
                st.write("Avec la régression logistique les deux méthodes BOW et Tf-Idf sont presque au même niveau en terme de précision et semblent être les plus précises dans la prédiction des classes.")
        
#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

        st.subheader("2. Random Forest (classification logistique multinominale)")


        columns_tert_class_rf = ["Matrices de confusion RF (classification logistique multinominale)", "Comparaison des méthodes de W-embedding pour RF (classification logistique multinominale)"]

        cm_tert_class_rf_chbox = st.checkbox(columns_tert_class_rf[0])
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        report_tert_class_rf_chbox = st.checkbox(columns_tert_class_rf[1])
        
        col_cm_tert_rf, col_report_tert_rf = st.columns(2)
        
        with col_cm_tert_rf:
            
            if cm_tert_class_rf_chbox:
                image_CM_teriary_class_rf = Image.open("figCMRandomforest.png")
                st.image(image_CM_teriary_class_rf, caption="Classification logistique multinominale: matrices de confusion pour différentes méthodes de word embedding avec Random Forest.")
                st.write()
                st.write("La classe neutre globalement prédite avec plus de 50% parmi les bonnes prédictions")
                st.write()
                st.write("Les méthodes BOW et Tf-Idf permettent également des bonnes prédictions de plus de 50% sur la classe négative (0) ")
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with col_report_tert_rf:
                
            if report_tert_class_rf_chbox:
                image_Report_teriary_class_rf = Image.open("figReportRandomForest.png")
                st.image(image_Report_teriary_class_rf, caption="Classification logistique multinominale: comparaison entre les méthodes de word embedding avec Random Forest.")
                st.write()
                st.write("La méthode Tf-Idf semble mieux prédire globalement avec le modèle Random Forest.")

        
#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
        st.subheader("3. XGBoost (classification logistique multinominale)")
       
        columns_binary_class_xgb = ["Matrices de confusion XGBoost (classification logistique multinominale)", "Comparaison des méthodes de W-embedding avec XGBoost (classification logistique multinominale)"]

        cm_tert_class_xgb_chbox = st.checkbox(columns_binary_class_xgb[0])
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        report_tert_class_xgb_chbox = st.checkbox(columns_binary_class_xgb[1])
        
        col_cm_tert_xgb, col_report_tert_xgb = st.columns(2)
        
        with col_cm_tert_xgb:
            
            if cm_tert_class_xgb_chbox:
                image_CM_teriary_class_xgb = Image.open("figCMXGBoost.png")
                st.image(image_CM_teriary_class_xgb, caption="Classification logistique multinominale: matrices de confusion pour différentes méthodes de word embedding avec XGBoost.")
                st.write()
                st.write("Résultats analogues aux précédents: la classe neutre globalement prédite avec plus de 50% parmi les bonnes prédictions")
                st.write()
                st.write("Les méthodes BOW et Tf-Idf permettent également des bonnes prédictions de plus de 50% sur la classe négative (0) ")
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with col_report_tert_xgb:
                
            if report_tert_class_xgb_chbox:
                image_Report_teriary_class_xgb = Image.open("figReportXGBoost.png")
                st.image(image_Report_teriary_class_xgb, caption="Classification logistique multinominale: comparaison entre les méthodes de word embedding avec Random Forest.")
                st.write()
                st.write("Meilleure précision (accuracy) globale par la méthode BOW avec le modèle de classification multinominale via XGBoost.")
        
        
        st.subheader("!!! Discussion sur la classification logistique multinominale à 3 classes")
        
        commentaire_tert_class_chbox = st.checkbox("Discussions sur la classification multinominale")

        col_multiclas_chart, col_multiclas_repartition_graph = st.columns(2)
        
        if commentaire_tert_class_chbox:
            st.write()
            
            with col_multiclas_chart:
                st.write(df_demo_multinominalclass.head())
                
                st.write()
                st.write("==> Précision sur la classification de la classe neutre généralement plus grande: elle est la plus représentée en effet;")
                
                st.write()
                st.write("==> Certaines notes de satisfaction pas strictement en accord avec le commentaires; ")
                
                st.write()
                st.write("==> Difficile de distinguer parfaitement la classe 'neutre' aux autres classes.")
            
            with col_multiclas_repartition_graph:
                   image_multiclass_repartion = Image.open("figmulticlassrepartition.png")
                   st.image(image_multiclass_repartion,use_column_width=True, caption="Répartion des classes pour la classification multinominale.")


        
#=================================================================================
#=================================================================================
        
        #if comment_classification_choice == "Classification binaire":
        st.header("B - Classification binaire ")

        st.write()
        st.write("==> Classification à 2 classes")
        
        st.write()
        st.write("==> Classe négative => label = 0")
        st.write()
        st.write("==> Classe positive => label = 1")
    
        st.write()
        st.write("==> Trois (3) modèles proposés: Régression logistique, Random Forest , XGBoost")
        
        st.write()
        st.write("Quatre méthodes d'embedding pour chaque modèle de W-E: Bag Of Words (BOW), Term Frequency - Inverse document Frequency (Tf-Idf), Word2Vec et Doc2Vec")
        
#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
        # Logistic regression
        
        st.subheader("1. Régression logistique (classification binaire)")
          
        columns_binary_class_lreg = ["Matrices de confusion pour la régression logistique (binary classification)", "Comparaison des méthodes de W-Embedding pour la régression logistique (binary classification)"]

        cm_binary_class_lreg_chbox = st.checkbox(columns_binary_class_lreg[0])
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        report_binary_class_lreg_chbox = st.checkbox(columns_binary_class_lreg[1])
        
        col_cm_binary_lreg, col_report_binary_lreg = st.columns(2)
        
        with col_cm_binary_lreg:
            
            if cm_binary_class_lreg_chbox:
                image_CM_binary_class_logreg = Image.open("figCMbinaryLogisticregression.png")
                st.image(image_CM_binary_class_logreg, caption="Classification binaire: matrices de confusion pour différentes méthodes de word embedding avec la régression logistique.")
                st.write()
                st.write("On obtient au moins 55% environs sur toutes les bonnes prédictions et avec toutes les méthodes de W-Embedding")
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with col_report_binary_lreg:
                
            if report_binary_class_lreg_chbox:
                 image_Report_binary_class_logreg = Image.open("figReportbinaryLogisticregression.png")
                 st.image(image_Report_binary_class_logreg, caption="Classification binaire: comparaison entre les méthodes de word embedding avec la régression logistique.")
                 st.write("Les deux méthodes BOW et Tf-Idf permettent globalement d'obtenir la précision la plus élevée")
#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
        #st.markdown("2. Random Forest (classification binaire)")
        st.subheader("2. Random Forest (classification binaire)")
          
        columns_binary_class_rf = ["Matrices de confusion avec Random forest (binary classification)", "Comparaison des méthodes de W-Embedding avec Random Forest (binary classification)"]
        
        cm_binary_class_rf_chbox = st.checkbox(columns_binary_class_rf[0])
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        report_binary_class_rf_chbox = st.checkbox(columns_binary_class_rf[1])
        
        col_cm_binary_rf, col_report_binary_rf = st.columns(2)
        
        with col_cm_binary_rf:
            
            if cm_binary_class_rf_chbox:
                image_CM_binary_class_rf = Image.open("figCMbinaryRandomforest.png")
                st.image(image_CM_binary_class_rf, caption="Classification binaire: matrices de confusion pour différentes méthodes de word embedding avec Random Forest.")
                st.write()
                st.write("Amélioration nette des pourcentages des bonnes prédictions (au moins 60%) pour toutes les méthodes")
        
                
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with col_report_binary_rf:
                
            if report_binary_class_rf_chbox:
                image_Report_binary_class_rf = Image.open("figReportbinaryRandomForest.png")
                st.image(image_Report_binary_class_rf, caption="Classification binaire: comparaison entre les méthodes de word embedding avec Random Forest.")
                st.write()
                st.write("Précision globale meilleure avec Tf-Idf sur la classification binaire avec Random Forest")
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°            #°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
        #st.markdown("3. XGBoost (classification binaire)")
        st.subheader("3. XGBoost (classification binaire)")
        
        
        columns_binary_class_xgb = ["Matrices de confusion avec XGBoost (binary classification)", "Comparaison des méthodes de W-Embedding avec XGBoost (binary classification)"]
        
        cm_binary_class_xgb_chbox = st.checkbox(columns_binary_class_xgb[0])
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        report_binary_class_xgb_chbox = st.checkbox(columns_binary_class_xgb[1])
        
        col_cm_binary_xgb, col_report_binary_xgb = st.columns(2)
        
        with col_cm_binary_xgb:
            
            if cm_binary_class_xgb_chbox:
                image_CM_binary_class_xgb = Image.open("figCMbinaryXGBoost.png")
                st.image(image_CM_binary_class_xgb, caption="Classification binaire: matrices de confusion pour différentes méthodes de word embedding avec XGBoost.")
                st.write()
                st.write("Egalement au minimum 60% de bonnes prédictions pour toutes les méthodes de Word embedding utilisées")
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with col_report_binary_xgb:
                
            if report_binary_class_xgb_chbox:
                image_Report_binary_class_xgb = Image.open("figReportbinaryXGBoost.png")
                st.image(image_Report_binary_class_xgb, caption="Classification binaire: comparaison entre les méthodes de word embedding avec XGBoost.")
                st.write()
                st.write("")
                
      
        st.subheader("!!! Discussion sur la classification binaire")
        
        commentaire_binary_class_chbox = st.checkbox("Discussions sur la classification binaire")
        
        df_demo_binary = pd.read_csv(filepath_or_buffer="base2021demo.csv")
        df_demo_binary = df_demo_binary.drop(columns=["Unnamed: 0"])
        

        col_binary_chart, col_binary_repartition_graph = st.columns(2)
        st.write()
        
        if commentaire_binary_class_chbox:
            with col_binary_chart:
                st.write(df_demo_binary.head())
                
                st.write()
                st.write("==> L'accuracy globale de prédiction de la classe positive est plus élevé globalement : en effet c'est la classe la plus représentée;")
                
                st.write()
                st.write("==> L'accuracy des bonnes prédictions généralement amélioré comparé à la classification multinominale. ")
                
            
            
            with col_binary_repartition_graph:
                image_binaryclassrepartion = Image.open("figbinaryclassrepartition.png")
                st.image(image_binaryclassrepartion,use_column_width=True, caption="Répartion des classes pour la classification binaire.")
            
# ===============================================================================    
#================================================================================
    elif choice == "Simulateur de la classification":
        st.title("Simulateur de la classification")
        
        # Charger les données (dataframe contenant les commentaires nettoyés)
        
        #df_demo_binary = pd.read_csv(filepath_or_buffer=filedirectory+"base2021demo.csv")

        
            # ===> Load pre-trained models
        rfclassfitmodel = load('rfClassPipeFitmodel.joblib') 
        encode_job = load('targetencoded.joblib') 

        
        # Nombre d'itérations des modèles
        num_estimators = 100
        #input_message = " Test" 
        
        st.header("A) Pre-trained model")
        #st.sidebar.checkbox('special')
        
        data_chbox = st.checkbox("Data and model")
        
        # if data_chbox:
        #     #uploaded_data_demo = st.file_uploader("Choose a demo data file")
        #     # if uploaded_data_demo is not None:
        #     #   df_demo = pd.read_csv(uploaded_data_demo)
        #     #   #st.write(dataframe)
        #     #   st.write(df_demo.head())
        #     #   #df_demo.head()
        #     #with st.expander("See data"):
        #     st.write(df_demo_binary.head())
        #     st.write()

        st.write()
        st.write("==> Modèle pré-entrainé: Random Forest;")
        st.write()
        st.write("==> Méthode de vectorisatoion: Term frequency - Inverse document frequency (Tf-Idf);")
        st.write()
        st.write(rfclassfitmodel)
            
        st.write()
        st.write("==> Le démonstrateur consiste à prédire la classe d'un commentaire \n quelconque à partir de notre modèle pré-entrainé.")
        
        st.write()
        st.header("B) Démonstrateur sur la classification binaire")
        demo_classif_binary_rf_chbox = st.checkbox("Demo classification binaire avec Random Forest")
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #print("here")
        # rfclassfitmodel = load('rfClassPipeFitmodel.joblib') 
        # encode_job = load('targetencoded.joblib') 
        
        
        if demo_classif_binary_rf_chbox:
            #st.write("==> La démonstration consiste à prédire la class d'un commentaire \n quelconque à partir de notre modèle pré-entrainé")
            #st.write()
            #st.write("==> Modèle Random Forest et la méthode TfIdf")


            #st.write("here")
            
            #input_message = "" 
            #st.write(rfclassfitmodel)
            
            def verbatim_filter(message) :
                #message = clean_text(message)
                arr_mess=np.array([message])
                result=encode_job.inverse_transform(rfclassfitmodel.predict(arr_mess))[0]
                #print("The message is : ", result)
                #st.write("The comment is classified as : ", result)
                return result
              
#             # Give a comment (as sentence)
            # input_message = " " 
    
            
            input_comment = st.text_input(label ="Entrer un commentaire en anglais:",value="",on_change=None)
            
            #if not name:
            if input_comment:
                
              #st.warning("Please fill out so required fields")
                input_message = input_comment
              
              # results
            #st.write()
                binary_class_result = verbatim_filter(input_message)
                
                col1, col2= st.columns(2)
                imwidth = 10
                
                if binary_class_result == "positive":
                    #st.write("GOOD !!!")
                    #width = st.slider('What is the width in pixels?', 0, 700, 350)
                                            
                    #with col1: 
                        #image_unlike = Image.open("hand_good.PNG")
                        #st.image(image_unlike,use_column_width=True, caption="Positive",  width=imwidth)
                    
                    with col1:
                            
                            st.write()
                            st.write("Ce commentaire est classifié comme : ** ", binary_class_result, " ** Il semble provenir d'un client très satisfait.")
                            image_like = Image.open("hand_good.png")
                            st.image(image_like,use_column_width=True, caption="Promoteur",  width=imwidth)
                            
    
                    with col2: 
                        image_like = Image.open("happy_customer.jpeg")
                        st.image(image_like,use_column_width=True, caption="Promoteur",  width=imwidth)
                        
                    #with col2:
                         
                         #st.write()
                         #st.write("Ce commentaire est classifié comme : ** ", binary_class_result, " ** Ce commentaire semble provenir d'un client Satisfait.")
                         #st.write(" Ce commentaire semble provenir d'un client satisfait.")
                        
                elif binary_class_result == "negative":
                    #st.write("GOOD !!!")
                   
    
                    #with col1: 
                        #image_unlike = Image.open("hand_bad.PNG")
                        #st.image(image_unlike,use_column_width=True, caption="Négative",  width=imwidth)
                        
                    with col1:
                         
                         st.write()
                         st.write("Ce commentaire est classifié comme : ** ", binary_class_result, " ** Ce commentaire semble provenir d'un client insatisfait.")
                         image_bad_hand = Image.open("hand_bad.png")
                         st.image(image_bad_hand,use_column_width=True, caption="Négative",  width=imwidth)
                         
                         
                    with col2: 
                        image_unlike = Image.open("unhappy_customer.jpeg")
                        st.image(image_unlike,use_column_width=True, caption="Detracteur",  width=imwidth)
                        
            else:
                
                    
                        #with st.expander("See the classification "):
                st.write("Vous n'avez pas entrer de commentaire !!!")
                    
                    
  ##################################################################################
##########################################################################################
###############################################################################################              


        
       
              
            #input_message = "I am not agree with the customer servic                   
# ===============================================================================    
#================================================================================

    elif choice == "Regard critique et perspectives":
        st.title("Regard critique et perspectives")
       
        st.write("==> Faire l'analyse par pays pour avoir plus de precision sur la recommandation et suprimer le Biais culturel qui peut etre dans les notation entre les pays")
        st.write()
        st.write("==> L'attente des clients est differente pour chaque business line il serait interessant de refaire l'analyse par business line : Banque, credit specialiste, Captive auto")
        st.write()
        st.write("==> Pour mesurer l'effet des plans d'action et l'evolution des drivers de la recommandation il serait interessant de faire l'analyse par année car l'exigence évolue dans le temps")
        st.write()
        st.write("==> Pour donner à l'organisme noté plus de visibilité sur ses performances future il serait interessant de faire un simulateur pour des previsions future des scores NPS")
        
    
    #================================================================================
    elif choice == "Conclusion":
        st.title("Conclusion")
       
        st.header("1. Exploration et modélisation du NPS")
        concl_explore_modeling_NPS_chbox = st.checkbox("L'essentiel sur l'exploratrion et modélisation du NPS")
       
        if concl_explore_modeling_NPS_chbox:
       
            st.write("==>  L'opinion qu'un client a de la marque pèse énormement dans l'explication de la recommandation, suivie par les offres de la marque, en troisième position on trouve les options de credit;")
            st.write()
           
            st.write("==> Regardant en details chaque dimension:")
            st.write()
            st.write("  -- à propos de l'image: ce qui favorise plus la recommandation c'est la confiance dans la marque, suivis par un critére d'actualité qui est la notion de la sustainbility ou le developpement durable;")
           
            st.write()
            st.write("  -- à propos l'offre: les clients souhaitent avoir des solutions au juste prix, des solutions flexibles et personnalisables;")
           
            st.write()
            st.write("  -- Les clients recommandent plus les marques avec qui ils partagent les meme valeurs;")
           
            st.write()
            st.write("  -- Les clients recommandent les marques qui leur donne le pouvoir de controler leur situation,")
           
            st.write()
            st.write("  -- Les clients recommandent les marques qui les traitent d'une manniére transparente est équitable")
           
            st.write()
            st.write("  -- Les clients recommandent les marques qui facilitent l'acces à leurs services et qui permettent aux clients de mieux profiter de la vie")

        st.header("2. Analyse des verbatims")
        concl_analyse_verbatims_chbox = st.checkbox("Important sur l'analyse des verbatims:")
       
        if concl_analyse_verbatims_chbox:
            st.write()
            st.write("==> Le taux d'interet et la facilité des process et des services sont des éléments recurrents;")

        st.header("3. Classification des commentaires")
        concl_classification_commentaires_chbox = st.checkbox("A retenir sur la classification des commentaires:")
        
        if concl_classification_commentaires_chbox:
            st.write()
            st.write("==> Trois modèles de classification mis en place: régression logistique, Random Forest et XGBoost;")


            st.write()
            st.write("==> Quatre méthodes d'Embedding utilisées: BOW, TfIdf, Word2Vec et Doc2Vec")
                                        
            st.write()
            st.write("==> Classification logistique multinominale à trois classes : moins performante sur les bonnes prédictions\n car difficile de faire la différence entre les passifs et les détracteurs d'une part et les promoteurs d'autre part;")
            
            st.write()
            st.write("==> Classification binaire : précision améliorée sur les bonnes prédictions;")
                                                
        
        st.write("-------------------------------------------------")
    
    #================================================================================
    elif choice == "Auteurs du projet":
        #image = Image.open('customerstatisfaction.jpg')
        #st.image(image, caption='')
        st.title("Auteurs du projet")
        st.write("Mohamed KRIM [learn more >] (www.linkedin.com/in/mkrim)")
        st.write("Esso-passi PALI [learn more >] (https://scholar.google.com/citations?view_op=list_works&hl=fr&user=rdg-xjYAAAAJ) ")




if __name__ == "__main__":
    main()

#================================================================================



#================================================================================
