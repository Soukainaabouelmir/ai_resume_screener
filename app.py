import streamlit as st
import pdfplumber
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from collections import Counter
from datetime import datetime
import base64
from io import BytesIO

# Imports pour ML/NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Configuration de la page
st.set_page_config(
    page_title="AI Resume Screener Pro", 
    page_icon="", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #2f8243;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-card {
        background-color: #fce621;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-card {
        background-color: #f0051a;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation NLTK (avec gestion d'erreur)
@st.cache_resource
def init_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except:
        return False

# ---- FONCTIONS UTILITAIRES ----

def extract_text_from_pdf(pdf_file):
    """Extraction de texte améliorée avec gestion d'erreurs"""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du PDF: {str(e)}")
        return ""
    return text

def preprocess_text(text):
    if not text:
        return ""
    
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text)
    
    try:
        stop_words = set(stopwords.words('french') + stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        
        return ' '.join(lemmatized_tokens)
    except:
        return text

def extract_skills(text, skill_categories):
    found_skills = {}
    
    for category, skills_list in skill_categories.items():
        found_skills[category] = []
        for skill in skills_list:
            if skill.lower() in text.lower():
                found_skills[category].append(skill)
    
    return found_skills

def extract_experience_years(text):
    patterns = [
        r'(\d+)\s*(?:ans?|years?)\s*(?:d\'expérience|experience|d\'exp)',
        r'(\d+)\+?\s*(?:ans?|years?)',
        r'(\d{4})\s*-\s*(\d{4})',  
        r'depuis\s*(\d{4})'
    ]
    
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            if isinstance(matches[0], tuple):
                for match in matches:
                    if len(match) == 2 and match[1].isdigit() and match[0].isdigit():
                        years.append(int(match[1]) - int(match[0]))
            else:
                years.extend([int(match) for match in matches if match.isdigit()])
    
    return max(years) if years else 0

def calculate_advanced_similarity(cv_text, job_text):
    
    cv_processed = preprocess_text(cv_text)
    job_processed = preprocess_text(job_text)
    
    if not cv_processed or not job_processed:
        return {"tfidf_score": 0, "word_overlap": 0, "semantic_score": 0}
    
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([cv_processed, job_processed])
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        tfidf_score = 0
    
    # 2. Overlap de mots-clés
    cv_words = set(cv_processed.split())
    job_words = set(job_processed.split())
    
    if len(job_words) > 0:
        word_overlap = len(cv_words.intersection(job_words)) / len(job_words)
    else:
        word_overlap = 0
    
    # 3. Score sémantique simplifié (basé sur les n-grammes)
    try:
        vectorizer_2gram = TfidfVectorizer(ngram_range=(2, 3), max_features=500)
        ngram_matrix = vectorizer_2gram.fit_transform([cv_processed, job_processed])
        semantic_score = cosine_similarity(ngram_matrix[0:1], ngram_matrix[1:2])[0][0]
    except:
        semantic_score = 0
    
    return {
        "tfidf_score": float(tfidf_score),
        "word_overlap": float(word_overlap),
        "semantic_score": float(semantic_score)
    }

def generate_recommendations(cv_text, job_text, similarity_scores, skills_match):
    """Génération de recommandations personnalisées"""
    recommendations = []
    
    # Score global
    avg_score = np.mean(list(similarity_scores.values()))
    
    if avg_score < 0.3:
        recommendations.append("🔴 **Score faible**: Votre CV nécessite une refonte majeure pour correspondre à cette offre.")
    elif avg_score < 0.6:
        recommendations.append("🟡 **Score moyen**: Quelques ajustements peuvent améliorer significativement votre candidature.")
    else:
        recommendations.append("🟢 **Bon score**: Votre profil correspond bien à l'offre !")
    
    # Recommandations basées sur les compétences
    missing_skills = []
    for category, skills in skills_match.items():
        if not skills:
            missing_skills.append(category)
    
    if missing_skills:
        recommendations.append(f"**Compétences à développer**: {', '.join(missing_skills)}")
    
    # Recommandations spécifiques
    if similarity_scores['word_overlap'] < 0.2:
        recommendations.append(" **Mots-clés**: Intégrez plus de termes spécifiques de l'offre d'emploi.")
    
    if similarity_scores['semantic_score'] < 0.3:
        recommendations.append(" **Contexte**: Ajoutez des descriptions détaillées de vos expériences pertinentes.")
    
    return recommendations

# ---- INTERFACE PRINCIPALE ----

def main():
    # En-tête
    st.markdown('<h1 class="main-header"> AI Resume Screener Pro</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialisation NLTK
    init_nltk()
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("Configuration")
        
        analysis_mode = st.selectbox(
            "Mode d'analyse:",
            ["Analyse Complète", "Analyse Rapide", "Focus Compétences"]
        )
        
        include_recommendations = st.checkbox("Inclure les recommandations", value=True)
        show_detailed_metrics = st.checkbox("Métriques détaillées", value=True)
        
        st.markdown("---")
        st.header(" À propos")
        st.info("""
        **AI Resume Screener Pro** utilise des techniques avancées de NLP pour analyser la compatibilité entre votre CV et une offre d'emploi.
        
        **Fonctionnalités:**
        - Analyse multi-dimensionnelle
        - Extraction automatique de compétences
        - Recommandations personnalisées
        - Visualisations interactives
        """)
    
    # Interface principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header(" Téléversement du CV")
        uploaded_cv = st.file_uploader(
            "Glissez-déposez votre CV (PDF)",
            type="pdf",
            help="Formats supportés: PDF uniquement"
        )
        
        if uploaded_cv:
            st.success(f" CV téléversé: {uploaded_cv.name}")
            
            # Aperçu du CV
            with st.expander("Aperçu du contenu"):
                cv_text = extract_text_from_pdf(uploaded_cv)
                if cv_text:
                    st.text_area("Texte extrait:", cv_text[:500] + "...", height=200, disabled=True)
                else:
                    st.error("Impossible d'extraire le texte du PDF")
    
    with col2:
        st.header(" Description du Poste")
        job_description = st.text_area(
            "Collez ici la description complète du poste:",
            height=300,
            help="Plus la description est détaillée, plus l'analyse sera précise"
        )
        
        if job_description:
            word_count = len(job_description.split())
            st.info(f" {word_count} mots détectés")
    
    # Définition des catégories de compétences
    skill_categories = {
        "Techniques": [
            "Python", "JavaScript", "Java", "C++", "React", "Node.js", "SQL", "MongoDB",
            "Docker", "Kubernetes", "AWS", "Azure", "Git", "Machine Learning", "AI",
            "Data Science", "TensorFlow", "PyTorch", "Pandas", "NumPy"
        ],
        "Gestion": [
            "Management", "Leadership", "Scrum", "Agile", "Project Management",
            "Team Management", "Budget", "Planning", "Coordination"
        ],
        "Communication": [
            "Presentation", "Public Speaking", "Writing", "Communication",
            "Negotiation", "Customer Service", "Training"
        ],
        "Langues": [
            "English", "French", "Spanish", "German", "Mandarin", "Bilingual", "Multilingual"
        ]
    }
    
    # Analyse principale
    if uploaded_cv and job_description and len(job_description.strip()) > 50:
        st.markdown("---")
        st.header(" Analyse en Cours...")
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extraction du texte
        status_text.text("Extraction du texte du CV...")
        progress_bar.progress(20)
        cv_text = extract_text_from_pdf(uploaded_cv)
        
        if not cv_text:
            st.error(" Impossible d'analyser le CV. Vérifiez le format du fichier.")
            return
        
        # Calcul des similarités
        status_text.text("Calcul des scores de similarité...")
        progress_bar.progress(50)
        similarity_scores = calculate_advanced_similarity(cv_text, job_description)
        
        # Extraction des compétences
        status_text.text("Analyse des compétences...")
        progress_bar.progress(70)
        skills_found = extract_skills(cv_text, skill_categories)
        job_skills = extract_skills(job_description, skill_categories)
        
        # Extraction de l'expérience
        status_text.text("Analyse de l'expérience...")
        progress_bar.progress(90)
        experience_years = extract_experience_years(cv_text)
        
        # Finalisation
        progress_bar.progress(100)
        status_text.text("Analyse terminée !")
        
        # Effacement de la barre de progression
        progress_bar.empty()
        status_text.empty()
        
        
        st.markdown("---")
        st.header("Résultats de l'Analyse")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = np.mean(list(similarity_scores.values()))
            st.metric("Score Global", f"{avg_score:.2%}", delta=f"{avg_score-0.5:.2%}")
        
        with col2:
            st.metric("Score TF-IDF", f"{similarity_scores['tfidf_score']:.2%}")
        
        with col3:
            total_skills = sum(len(skills) for skills in skills_found.values())
            st.metric("Compétences Trouvées", total_skills)
        
        with col4:
            st.metric("Années d'Expérience", f"{experience_years} ans")
        
        # Graphiques de visualisation
        if show_detailed_metrics:
            st.markdown("###  Métriques Détaillées")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Graphique radar des scores
                categories = list(similarity_scores.keys())
                values = list(similarity_scores.values())
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=[cat.replace('_', ' ').title() for cat in categories],
                    fill='toself',
                    name='Scores',
                    line_color='rgb(31, 119, 180)'
                ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    title="Scores de Correspondance"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                skill_data = []
                for category, skills in skills_found.items():
                    skill_data.append({
                        'Catégorie': category,
                        'CV': len(skills),
                        'Offre': len(job_skills.get(category, []))
                    })
                
                df_skills = pd.DataFrame(skill_data)
                
                fig_skills = go.Figure()
                fig_skills.add_trace(go.Bar(
                    x=df_skills['Catégorie'],
                    y=df_skills['CV'],
                    name='Dans votre CV',
                    marker_color='lightblue'
                ))
                fig_skills.add_trace(go.Bar(
                    x=df_skills['Catégorie'],
                    y=df_skills['Offre'],
                    name='Dans l\'offre',
                    marker_color='orange'
                ))
                
                fig_skills.update_layout(
                    title="Compétences par Catégorie",
                    barmode='group',
                    xaxis_title="Catégories",
                    yaxis_title="Nombre de compétences"
                )
                
                st.plotly_chart(fig_skills, use_container_width=True)
        
        st.markdown("###  Analyse des Compétences")
        
        for category, skills in skills_found.items():
            if skills:
                with st.expander(f" {category} ({len(skills)} trouvées)"):
                    st.write(", ".join(skills))
            else:
                with st.expander(f"{category} (aucune trouvée)"):
                    job_cat_skills = job_skills.get(category, [])
                    if job_cat_skills:
                        st.write(f"**Compétences demandées:** {', '.join(job_cat_skills)}")
                    else:
                        st.write("Aucune compétence spécifique détectée dans l'offre.")
        
        if include_recommendations:
            st.markdown("###  Recommandations Personnalisées")
            
            recommendations = generate_recommendations(
                cv_text, job_description, similarity_scores, skills_found
            )
            
            for i, recommendation in enumerate(recommendations):
                if "🔴" in recommendation:
                    st.markdown(f'<div class="danger-card">{recommendation}</div>', unsafe_allow_html=True)
                elif "🟡" in recommendation:
                    st.markdown(f'<div class="warning-card">{recommendation}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="success-card">{recommendation}</div>', unsafe_allow_html=True)
        
        # Section de téléchargement du rapport
        st.markdown("###  Télécharger le Rapport")
        
        # Génération du rapport en format texte
        report = f"""
RAPPORT D'ANALYSE CV - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

SCORES GLOBAUX:
- Score Global: {avg_score:.2%}
- Score TF-IDF: {similarity_scores['tfidf_score']:.2%}
- Overlap Mots-clés: {similarity_scores['word_overlap']:.2%}
- Score Sémantique: {similarity_scores['semantic_score']:.2%}

COMPÉTENCES DÉTECTÉES:
"""
        
        for category, skills in skills_found.items():
            report += f"\n{category}: {', '.join(skills) if skills else 'Aucune'}"
        
        report += f"\n\nEXPÉRIENCE: {experience_years} années détectées"
        
        report += "\n\nRECOMMANDATIONS:\n"
        for rec in recommendations:
            # Nettoyage des emojis pour le rapport texte
            clean_rec = re.sub(r'[🔴🟡🟢]', '', rec)
            report += f"- {clean_rec}\n"
        
        # Bouton de téléchargement
        st.download_button(
            label="Télécharger le rapport complet",
            data=report,
            file_name=f"rapport_cv_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    elif uploaded_cv and job_description:
        st.warning(" La description du poste doit contenir au moins 50 caractères pour une analyse pertinente.")
    
    else:
        st.info(" Veuillez téléverser votre CV et fournir une description de poste pour commencer l'analyse.")

if __name__ == "__main__":
    main()