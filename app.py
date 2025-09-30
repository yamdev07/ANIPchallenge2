import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile
import easyocr
from deepface import DeepFace
import time
import os
import pandas as pd
from datetime import datetime
import urllib.request
import requests
import sqlite3
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(
    page_title="Système Avancé de Reconnaissance Faciale",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé étendu
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
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    .webcam-container {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 10px;
        background-color: #f8f9fa;
    }
    .age-display {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(45deg, #e3f2fd, #bbdefb);
        border-radius: 10px;
        margin: 1rem 0;
    }
    .history-item {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
    .stat-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .progress-container {
        margin: 0.5rem 0;
    }
    .progress-label {
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🔍 Système Avancé de Reconnaissance Faciale & OCR</h1>', unsafe_allow_html=True)


# -----------------------------
# 1️⃣ Initialisation de la base de données
# -----------------------------

def init_database():
    """Initialise la base de données SQLite"""
    conn = sqlite3.connect('face_analysis.db', check_same_thread=False)
    c = conn.cursor()

    # Table pour les analyses faciales
    c.execute('''
        CREATE TABLE IF NOT EXISTS face_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age INTEGER,
            gender TEXT,
            emotion TEXT,
            race TEXT,
            confidence REAL,
            image_path TEXT,
            analysis_type TEXT
        )
    ''')

    # Table pour les analyses OCR
    c.execute('''
        CREATE TABLE IF NOT EXISTS ocr_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            text_content TEXT,
            confidence REAL,
            text_count INTEGER,
            doc_type TEXT,
            image_path TEXT
        )
    ''')

    # Table pour les comparaisons de visages
    c.execute('''
        CREATE TABLE IF NOT EXISTS face_comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            similarity REAL,
            verified INTEGER,
            model_used TEXT,
            image1_path TEXT,
            image2_path TEXT
        )
    ''')

    conn.commit()
    return conn


# Initialisation de la DB
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = init_database()


# -----------------------------
# 2️⃣ Fonctions de gestion de la base de données
# -----------------------------

def save_face_analysis(age, gender, emotion, race, confidence, image_path=None):
    """Sauvegarde une analyse faciale dans la base de données"""
    try:
        conn = st.session_state.db_conn
        c = conn.cursor()

        c.execute('''
            INSERT INTO face_analyses 
            (timestamp, age, gender, emotion, race, confidence, image_path, analysis_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            age,
            gender,
            emotion,
            race,
            confidence,
            image_path,
            "Reconnaissance faciale"
        ))

        conn.commit()
        return True
    except Exception as e:
        st.error(f"Erreur sauvegarde analyse faciale: {e}")
        return False


def save_ocr_analysis(text_content, confidence, text_count, doc_type, image_path=None):
    """Sauvegarde une analyse OCR dans la base de données"""
    try:
        conn = st.session_state.db_conn
        c = conn.cursor()

        c.execute('''
            INSERT INTO ocr_analyses 
            (timestamp, text_content, confidence, text_count, doc_type, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            text_content,
            confidence,
            text_count,
            doc_type,
            image_path
        ))

        conn.commit()
        return True
    except Exception as e:
        st.error(f"Erreur sauvegarde OCR: {e}")
        return False


def save_face_comparison(similarity, verified, model_used, image1_path, image2_path):
    """Sauvegarde une comparaison de visages dans la base de données"""
    try:
        conn = st.session_state.db_conn
        c = conn.cursor()

        c.execute('''
            INSERT INTO face_comparisons 
            (timestamp, similarity, verified, model_used, image1_path, image2_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            similarity,
            int(verified),
            model_used,
            image1_path,
            image2_path
        ))

        conn.commit()
        return True
    except Exception as e:
        st.error(f"Erreur sauvegarde comparaison: {e}")
        return False


def get_analysis_history(limit=50):
    """Récupère l'historique des analyses"""
    try:
        conn = st.session_state.db_conn
        # Analyses faciales
        face_df = pd.read_sql('SELECT * FROM face_analyses ORDER BY timestamp DESC LIMIT ?', conn, params=(limit,))
        # Analyses OCR
        ocr_df = pd.read_sql('SELECT * FROM ocr_analyses ORDER BY timestamp DESC LIMIT ?', conn, params=(limit,))
        # Comparaisons
        comp_df = pd.read_sql('SELECT * FROM face_comparisons ORDER BY timestamp DESC LIMIT ?', conn, params=(limit,))

        return face_df, ocr_df, comp_df
    except Exception as e:
        st.error(f"Erreur récupération historique: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# -----------------------------
# 3️⃣ Fonctions de téléchargement automatique
# -----------------------------

def download_with_fallback(urls, filename):
    """Tente de télécharger depuis plusieurs URLs de secours"""
    for i, url in enumerate(urls):
        try:
            st.info(f"📥 Tentative {i + 1}/{len(urls)}: Téléchargement de {filename}...")

            if url.startswith('http'):
                urllib.request.urlretrieve(url, filename)
            else:
                st.warning(f"URL non valide: {url}")
                continue

            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                st.success(f"✅ {filename} téléchargé avec succès")
                return True
            else:
                if os.path.exists(filename):
                    os.remove(filename)

        except Exception as e:
            st.warning(f"❌ Échec avec l'URL {i + 1}: {str(e)}")
            continue

    st.error(f"❌ Impossible de télécharger {filename} après {len(urls)} tentatives")
    return False


# -----------------------------
# 4️⃣ Chargement des modèles
# -----------------------------

@st.cache_resource
def load_models():
    """Charge tous les modèles nécessaires avec gestion d'erreurs"""
    models = {}

    # Vérification et téléchargement des fichiers AgeNet
    AGE_PROTOTXT = "deploy_age.prototxt"
    AGE_CAFFEMODEL = "age_net.caffemodel"

    # URLs alternatives pour les modèles AgeNet
    age_prototxt_urls = [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    ]

    age_caffemodel_urls = [
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
    ]

    # Téléchargement seulement si les fichiers n'existent pas
    age_net_available = False

    if not os.path.exists(AGE_PROTOTXT):
        download_with_fallback(age_prototxt_urls, AGE_PROTOTXT)

    if not os.path.exists(AGE_CAFFEMODEL):
        download_with_fallback(age_caffemodel_urls, AGE_CAFFEMODEL)

    # Vérification finale de la disponibilité des fichiers
    if os.path.exists(AGE_PROTOTXT) and os.path.exists(AGE_CAFFEMODEL):
        try:
            prototxt_size = os.path.getsize(AGE_PROTOTXT)
            caffemodel_size = os.path.getsize(AGE_CAFFEMODEL)

            if prototxt_size > 1000 and caffemodel_size > 1000000:
                models['age_net'] = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT, AGE_CAFFEMODEL)
                age_net_available = True
                st.success("✅ Modèle AgeNet chargé avec succès")
            else:
                st.warning("⚠️ Fichiers AgeNet corrompus ou incomplets")
                if os.path.exists(AGE_PROTOTXT):
                    os.remove(AGE_PROTOTXT)
                if os.path.exists(AGE_CAFFEMODEL):
                    os.remove(AGE_CAFFEMODEL)

        except Exception as e:
            st.error(f"❌ Erreur lors du chargement d'AgeNet: {e}")
    else:
        st.warning("⚠️ Fichiers AgeNet introuvables. Utilisation de DeepFace pour l'âge.")

    # Chargement du détecteur de visages
    try:
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        ]

        for cascade_path in cascade_paths:
            if os.path.exists(cascade_path):
                models['face_cascade'] = cv2.CascadeClassifier(cascade_path)
                st.success("✅ Détecteur de visages chargé avec succès")
                break
        else:
            st.error("❌ Aucun détecteur de visages trouvé")

    except Exception as e:
        st.error(f"❌ Erreur chargement détecteur visages: {e}")

    # Chargement OCR
    try:
        models['ocr_reader'] = easyocr.Reader(['en', 'fr'])
        st.success("✅ Modèle OCR chargé avec succès")
    except Exception as e:
        st.error(f"❌ Erreur chargement OCR: {e}")

    return models, age_net_available


# -----------------------------
# 5️⃣ Initialisation des modèles
# -----------------------------

with st.spinner("🔧 Initialisation des modèles en cours..."):
    models, age_net_available = load_models()


# -----------------------------
# 6️⃣ Fonctions utilitaires améliorées
# -----------------------------

def cleanup_temp_files():
    """Nettoie les fichiers temporaires potentiellement bloqués"""
    import tempfile
    import gc

    # Forcer le garbage collection
    gc.collect()

    # Nettoyer le répertoire temporaire
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.startswith('tmp') and (file.endswith('.jpg') or file.endswith('.png')):
            try:
                file_path = os.path.join(temp_dir, file)
                os.remove(file_path)
            except PermissionError:
                # Ignorer les fichiers encore utilisés
                pass


# Nettoyage au démarrage
if 'cleaned_up' not in st.session_state:
    cleanup_temp_files()
    st.session_state.cleaned_up = True


def safe_deepface_analysis(image, actions=['age', 'gender', 'emotion', 'race']):
    """
    Version sécurisée de DeepFace.analyze qui évite les problèmes de fichiers
    """
    try:
        if isinstance(image, str):
            # Si c'est un chemin de fichier
            return DeepFace.analyze(
                img_path=image,
                actions=actions,
                enforce_detection=False,
                detector_backend='opencv'
            )
        else:
            # Si c'est une image PIL ou numpy array
            if isinstance(image, Image.Image):
                img_array = np.array(image.convert('RGB'))
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = image

            return DeepFace.analyze(
                img_path=img_bgr,
                actions=actions,
                enforce_detection=False,
                detector_backend='opencv'
            )
    except Exception as e:
        st.error(f"Erreur analyse DeepFace: {e}")
        return None


def safe_progress(value, key=None):
    """Version sécurisée de st.progress qui gère les valeurs hors limites"""
    try:
        # S'assurer que la valeur est entre 0 et 1
        normalized_value = max(0.0, min(1.0, float(value)))
        st.progress(normalized_value, key=key)
    except (ValueError, TypeError) as e:
        st.warning(f"Valeur de progression invalide: {value}")


def estimate_real_age(image):
    """Estimation d'âge réel améliorée avec multiples méthodes"""
    try:
        img = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if 'face_cascade' not in models:
            return None, "Détecteur de visages non disponible", None

        faces = models['face_cascade'].detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return None, "Aucun visage détecté", None

        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]

        # ✅ CORRECTION: Utiliser l'image en mémoire au lieu de fichier temporaire
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Méthode 1: DeepFace avec image en mémoire
        try:
            result = safe_deepface_analysis(img_bgr, actions=['age'])

            if result and len(result) > 0:
                real_age = int(result[0]['age'])
                confidence = 0.8

                # Méthode 2: AgeNet
                if age_net_available and 'age_net' in models:
                    try:
                        face_img = img[y:y + h, x:x + w]
                        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                                     (78.4263377603, 87.7689143744, 114.895847746),
                                                     swapRB=False)
                        models['age_net'].setInput(blob)
                        preds = models['age_net'].forward()

                        age_ranges = [(0, 2), (4, 6), (8, 12), (15, 20),
                                      (25, 32), (38, 43), (48, 53), (60, 100)]
                        age_net_pred = age_ranges[preds[0].argmax()]
                        age_net_confidence = preds[0].max()

                        age_net_mid = sum(age_net_pred) / 2
                        final_age = int((real_age * 0.7) + (age_net_mid * 0.3))
                        confidence = (confidence * 0.7) + (age_net_confidence * 0.3)

                        return final_age, confidence, (x, y, w, h)

                    except Exception as e:
                        st.warning(f"AgeNet échec, utilisation DeepFace seul: {e}")

                return real_age, confidence, (x, y, w, h)

        except Exception as e:
            st.warning(f"DeepFace échec: {e}")

        # Méthode 3: Fallback AgeNet
        if age_net_available and 'age_net' in models:
            try:
                face_img = img[y:y + h, x:x + w]
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                             (78.4263377603, 87.7689143744, 114.895847746),
                                             swapRB=False)
                models['age_net'].setInput(blob)
                preds = models['age_net'].forward()

                age_ranges = [(0, 2), (4, 6), (8, 12), (15, 20),
                              (25, 32), (38, 43), (48, 53), (60, 100)]
                best_idx = preds[0].argmax()
                age_range = age_ranges[best_idx]
                confidence = preds[0][best_idx]

                real_age = int(sum(age_range) / 2)
                return real_age, confidence, (x, y, w, h)

            except Exception as e:
                return None, f"Erreur AgeNet: {str(e)}", None

        return None, "Aucune méthode d'estimation d'âge disponible", None

    except Exception as e:
        return None, f"Erreur: {str(e)}", None


def enhanced_ocr_analysis(image):
    """OCR amélioré avec analyse de contexte"""
    try:
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = image

        if 'ocr_reader' not in models:
            return "OCR non disponible", 0, 0

        results = models['ocr_reader'].readtext(image_np, detail=1, paragraph=False)

        texts = []
        confidences = []

        for (bbox, text, confidence) in results:
            if confidence > 0.3:
                texts.append(text)
                confidences.append(confidence)

        avg_confidence = np.mean(confidences) if confidences else 0
        full_text = "\n".join(texts) if texts else "Aucun texte détecté"

        return full_text, avg_confidence, len(texts)

    except Exception as e:
        return f"Erreur OCR: {str(e)}", 0, 0


def advanced_face_comparison(img1, img2):
    """Comparaison de visages avancée avec multiple modèles"""
    try:
        models_to_try = ['VGG-Face', 'Facenet', 'ArcFace', 'OpenFace']
        results = []

        for model_name in models_to_try:
            try:
                # ✅ Utiliser les images directement sans fichiers temporaires
                result = DeepFace.verify(
                    img1_path=img1,
                    img2_path=img2,
                    model_name=model_name,
                    distance_metric='cosine',
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                results.append({
                    'model': model_name,
                    'verified': result['verified'],
                    'distance': result['distance'],
                    'similarity': (1 - result['distance']) * 100
                })
            except Exception as e:
                st.warning(f"Modèle {model_name} échoué: {e}")
                continue

        if not results:
            return False, 1.0, 0.0, "Aucun modèle n'a fonctionné"

        best_result = max(results, key=lambda x: x['similarity'])

        if best_result['distance'] <= 0.4:
            confidence_level = "Élevée"
        elif best_result['distance'] <= 0.6:
            confidence_level = "Moyenne"
        else:
            confidence_level = "Faible"

        return (best_result['verified'],
                best_result['distance'],
                best_result['similarity'],
                f"{best_result['model']} (Confiance: {confidence_level})")

    except Exception as e:
        return False, 1.0, 0.0, f"Erreur: {str(e)}"


# -----------------------------
# 7️⃣ Fonctions pour le mode temps réel
# -----------------------------

class WebcamProcessor:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.current_frame = None

    def start_webcam(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                st.error("❌ Impossible d'accéder à la webcam")
                return False
            self.is_running = True
            return True
        except Exception as e:
            st.error(f"❌ Erreur lors du démarrage de la webcam: {e}")
            return False

    def stop_webcam(self):
        self.is_running = False
        if self.cap:
            self.cap.release()

    def get_frame(self):
        if self.cap and self.is_running:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = frame_rgb
                return frame_rgb
        return None


if 'webcam_processor' not in st.session_state:
    st.session_state.webcam_processor = WebcamProcessor()


# -----------------------------
# 8️⃣ Nouvelle section: Statistiques et Historique
# -----------------------------

def show_statistics():
    """Affiche les statistiques des analyses"""
    st.markdown("## 📊 Tableau de bord statistique")

    face_df, ocr_df, comp_df = get_analysis_history(1000)

    if not face_df.empty:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_age = face_df['age'].mean()
            st.markdown(f'<div class="stat-card">🎂 Âge moyen<br><h3>{avg_age:.1f} ans</h3></div>',
                        unsafe_allow_html=True)

        with col2:
            total_analyses = len(face_df)
            st.markdown(f'<div class="stat-card">🔍 Analyses totales<br><h3>{total_analyses}</h3></div>',
                        unsafe_allow_html=True)

        with col3:
            success_rate = (face_df['confidence'] > 0.5).mean() * 100
            st.markdown(f'<div class="stat-card">✅ Taux de succès<br><h3>{success_rate:.1f}%</h3></div>',
                        unsafe_allow_html=True)

        with col4:
            if not ocr_df.empty:
                avg_text_count = ocr_df['text_count'].mean()
                st.markdown(f'<div class="stat-card">📖 Mots moyens<br><h3>{avg_text_count:.1f}</h3></div>',
                            unsafe_allow_html=True)

        # Graphique de distribution des âges
        st.markdown("### 📈 Distribution des âges analysés")
        if len(face_df) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(face_df['age'].dropna(), bins=20, kde=True, ax=ax)
            ax.set_xlabel('Âge')
            ax.set_ylabel('Nombre d\'analyses')
            ax.set_title('Distribution des âges détectés')
            st.pyplot(fig)

        # Graphique des émotions
        st.markdown("### 😊 Répartition des émotions")
        if 'emotion' in face_df.columns:
            emotion_counts = face_df['emotion'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            emotion_counts.plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('')
            st.pyplot(fig)


def show_history():
    """Affiche l'historique des analyses"""
    st.markdown("## 📜 Historique des analyses")

    face_df, ocr_df, comp_df = get_analysis_history(50)

    tab1, tab2, tab3 = st.tabs(["🧠 Analyses faciales", "📄 Analyses OCR", "🔍 Comparaisons"])

    with tab1:
        if not face_df.empty:
            for _, analysis in face_df.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="history-item">
                        <strong>📅 {analysis['timestamp'][:16]}</strong><br>
                        🎂 Âge: {analysis['age']} ans | 👤 Genre: {analysis['gender']}<br>
                        😊 Emotion: {analysis['emotion']} | 🌍 Origine: {analysis['race']}<br>
                        📊 Confiance: {analysis['confidence']:.2f}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Aucune analyse faciale dans l'historique")

    with tab2:
        if not ocr_df.empty:
            for _, analysis in ocr_df.iterrows():
                with st.container():
                    text_preview = analysis['text_content'][:100] + "..." if len(analysis['text_content']) > 100 else \
                    analysis['text_content']
                    st.markdown(f"""
                    <div class="history-item">
                        <strong>📅 {analysis['timestamp'][:16]}</strong><br>
                        📖 Texte: {text_preview}<br>
                        🔤 Mots: {analysis['text_count']} | 📊 Confiance: {analysis['confidence']:.2f}<br>
                        📄 Type: {analysis['doc_type']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Aucune analyse OCR dans l'historique")

    with tab3:
        if not comp_df.empty:
            for _, analysis in comp_df.iterrows():
                with st.container():
                    status = "✅ Correspondance" if analysis['verified'] else "❌ Non correspondance"
                    st.markdown(f"""
                    <div class="history-item">
                        <strong>📅 {analysis['timestamp'][:16]}</strong><br>
                        {status}<br>
                        📈 Similarité: {analysis['similarity']:.1f}%<br>
                        🤖 Modèle: {analysis['model_used']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Aucune comparaison dans l'historique")


# -----------------------------
# 9️⃣ Navigation principale
# -----------------------------

# Menu de navigation
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio("", ["🏠 Analyse en direct", "📊 Statistiques", "📜 Historique"])

# -----------------------------
# 🔟 Page: Analyse en direct
# -----------------------------

if page == "🏠 Analyse en direct":
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🔧 Modèles disponibles")
        if age_net_available:
            st.success("✅ AgeNet: Disponible")
        else:
            st.warning("⚠️ AgeNet: Non disponible")

        if 'face_cascade' in models:
            st.success("✅ Détecteur visages: Disponible")
        else:
            st.error("❌ Détecteur visages: Non disponible")

        if 'ocr_reader' in models:
            st.success("✅ OCR: Disponible")
        else:
            st.error("❌ OCR: Non disponible")

        st.subheader("🎯 Paramètres")
        confidence_threshold = st.slider("Seuil de confiance", 0.1, 1.0, 0.5, 0.1)

    # Interface principale d'analyse
    col1, col2 = st.columns([2, 1])

    with col1:
        mode = st.radio("**Mode d'opération:**",
                        ["📁 Upload d'image", "📷 Temps réel (Webcam)"],
                        horizontal=True)

        uploaded_file = None
        image_path = None
        image = None

        if mode == "📁 Upload d'image":
            uploaded_file = st.file_uploader("Choisissez une image",
                                             type=["jpg", "jpeg", "png", "bmp"])

            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
                    temp.write(uploaded_file.read())
                    image_path = temp.name
                image = Image.open(image_path)

        else:
            st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
            st.subheader("🎥 Mode Temps Réel")

            col_web1, col_web2 = st.columns(2)
            with col_web1:
                if st.button("🚀 Démarrer la webcam"):
                    if st.session_state.webcam_processor.start_webcam():
                        st.success("✅ Webcam démarrée")
            with col_web2:
                if st.button("🛑 Arrêter la webcam"):
                    st.session_state.webcam_processor.stop_webcam()
                    st.info("⏹️ Webcam arrêtée")

            if st.session_state.webcam_processor.is_running:
                webcam_placeholder = st.empty()
                if st.button("📸 Capturer"):
                    frame = st.session_state.webcam_processor.current_frame
                    if frame is not None:
                        image = Image.fromarray(frame)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
                            image.save(temp.name)
                            image_path = temp.name
                        st.success("✅ Image capturée!")

                # Simulation temps réel
                frame = st.session_state.webcam_processor.get_frame()
                if frame is not None:
                    webcam_placeholder.image(frame, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("🎯 Fonctionnalités")
        capabilities = [
            "🔍 Reconnaissance faciale", "🎂 Estimation d'âge RÉEL",
            "📖 OCR intelligent", "🔄 Comparaison de visages",
            "😊 Analyse des émotions", "👤 Détection du genre"
        ]
        for cap in capabilities:
            st.write(f"• {cap}")

    # Sélection de la tâche
    task = st.radio("**Tâche à effectuer:**",
                    ["🧠 Reconnaissance faciale & âge",
                     "🔍 Matching de visages",
                     "📄 OCR & Analyse de document"],
                    horizontal=True)

    # Traitement des tâches
    if (mode == "📁 Upload d'image" and uploaded_file is not None) or (
            mode == "📷 Temps réel (Webcam)" and image is not None):
        if task == "🧠 Reconnaissance faciale & âge":
            st.subheader("🧠 Résultats de l'analyse faciale")

            with st.spinner("🔍 Analyse du visage en cours..."):
                real_age, confidence, bbox = estimate_real_age(image)
                deepface_result = safe_deepface_analysis(image)

            if real_age is not None and deepface_result and len(deepface_result) > 0:
                # Sauvegarde dans la base de données
                df_data = deepface_result[0]
                save_face_analysis(
                    real_age,
                    df_data.get('dominant_gender', 'Inconnu'),
                    df_data.get('dominant_emotion', 'Neutre'),
                    df_data.get('dominant_race', 'Inconnue'),
                    confidence if confidence else 0.5,
                    image_path
                )

                # Affichage des résultats
                col_res1, col_res2 = st.columns(2)

                with col_res1:
                    st.markdown(f'<div class="age-display">🎂 Âge estimé: {real_age} ans</div>', unsafe_allow_html=True)

                    # Affichage de l'image avec bounding box
                    if bbox:
                        img_with_bbox = np.array(image.convert('RGB'))
                        x, y, w, h = bbox
                        cv2.rectangle(img_with_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        st.image(img_with_bbox, caption="Visage détecté", use_container_width=True)
                    else:
                        st.image(image, caption="Image analysée", use_container_width=True)

                with col_res2:
                    st.markdown("### 📊 Détails de l'analyse")

                    # Informations démographiques
                    st.markdown(f"**👤 Genre:** {df_data.get('dominant_gender', 'Inconnu')}")
                    st.markdown(f"**😊 Émotion dominante:** {df_data.get('dominant_emotion', 'Inconnue')}")
                    st.markdown(f"**🌍 Origine estimée:** {df_data.get('dominant_race', 'Inconnue')}")
                    st.markdown(
                        f"**📊 Niveau de confiance:** {confidence:.2%}" if confidence else "**📊 Niveau de confiance:** Non disponible")

                    # Détails des émotions
                    if 'emotion' in df_data:
                        st.markdown("#### 📈 Distribution des émotions:")
                        emotions = df_data['emotion']
                        for emotion, score in emotions.items():
                            progress = int(score * 100)
                            st.markdown(f"**{emotion}:** {progress}%")
                            # ✅ CORRECTION: Utilisation de safe_progress
                            safe_progress(score, key=f"emotion_{emotion}")

                st.success("✅ Analyse terminée avec succès!")

            else:
                st.error("❌ Impossible d'analyser le visage. Veuillez essayer avec une autre image.")

        elif task == "🔍 Matching de visages":
            st.subheader("🔍 Comparaison de visages")

            col_comp1, col_comp2 = st.columns(2)

            with col_comp1:
                st.markdown("**Image de référence:**")
                ref_file = st.file_uploader("Choisissez l'image de référence",
                                            type=["jpg", "jpeg", "png", "bmp"],
                                            key="ref")
                if ref_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_ref:
                        temp_ref.write(ref_file.read())
                        ref_path = temp_ref.name
                    st.image(ref_path, caption="Image de référence", use_container_width=True)

            with col_comp2:
                st.markdown("**Image à comparer:**")
                if image:
                    st.image(image, caption="Image à comparer", use_container_width=True)

            if 'ref_path' in locals() and image:
                if st.button("🔍 Lancer la comparaison"):
                    with st.spinner("🔍 Comparaison des visages en cours..."):
                        # Convertir l'image PIL en format compatible
                        img_bgr = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)

                        verified, distance, similarity, model_info = advanced_face_comparison(ref_path, img_bgr)

                        # Sauvegarde
                        save_face_comparison(similarity, verified, model_info, ref_path, image_path)

                        # Affichage résultats
                        col_res1, col_res2 = st.columns(2)

                        with col_res1:
                            if verified:
                                st.markdown('<div class="success-box">✅ **CORRESPONDANCE TROUVÉE**</div>',
                                            unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="error-box">❌ **AUCUNE CORRESPONDANCE**</div>',
                                            unsafe_allow_html=True)

                            st.markdown(f"**📈 Score de similarité:** {similarity:.2f}%")
                            st.markdown(f"**📏 Distance:** {distance:.4f}")
                            st.markdown(f"**🤖 Modèle utilisé:** {model_info}")

                        with col_res2:
                            # Barre de similarité
                            st.markdown(f"**Niveau de confiance:**")
                            # ✅ CORRECTION: Utilisation de safe_progress
                            similarity_progress = similarity / 100
                            safe_progress(similarity_progress, key="similarity")

                            if similarity >= 80:
                                st.success("🔒 Forte probabilité de correspondance")
                            elif similarity >= 60:
                                st.warning("⚠️ Correspondance modérée")
                            else:
                                st.error("🔓 Faible probabilité de correspondance")

        elif task == "📄 OCR & Analyse de document":
            st.subheader("📄 Analyse OCR du document")

            if image:
                with st.spinner("🔍 Analyse du texte en cours..."):
                    text_content, avg_confidence, text_count = enhanced_ocr_analysis(image)

                # Déterminer le type de document
                doc_type = "Document"
                if text_count > 200:
                    doc_type = "Document long"
                elif "facture" in text_content.lower() or "invoice" in text_content.lower():
                    doc_type = "Facture"
                elif "carte" in text_content.lower() or "id" in text_content.lower():
                    doc_type = "Carte d'identité"
                elif "passeport" in text_content.lower():
                    doc_type = "Passeport"

                # Sauvegarde
                save_ocr_analysis(text_content, avg_confidence, text_count, doc_type, image_path)

                # Affichage résultats
                col_ocr1, col_ocr2 = st.columns([1, 1])

                with col_ocr1:
                    st.image(image, caption="Document analysé", use_container_width=True)

                with col_ocr2:
                    st.markdown("### 📊 Résultats OCR")
                    st.markdown(f"**📄 Type de document:** {doc_type}")
                    st.markdown(f"**🔤 Nombre de mots détectés:** {text_count}")
                    st.markdown(f"**📊 Confiance moyenne:** {avg_confidence:.2%}")

                    st.markdown("#### 📝 Texte extrait:")
                    st.text_area("Texte", text_content, height=200)

                st.success(f"✅ Analyse OCR terminée! {text_count} mots détectés.")

# -----------------------------
# 📊 Page: Statistiques
# -----------------------------

elif page == "📊 Statistiques":
    show_statistics()

# -----------------------------
# 📜 Page: Historique
# -----------------------------

elif page == "📜 Historique":
    show_history()

    # Options d'export
    st.markdown("### 💾 Export des données")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        if st.button("📊 Exporter en CSV"):
            face_df, ocr_df, comp_df = get_analysis_history(10000)
            if not face_df.empty:
                csv = face_df.to_csv(index=False)
                st.download_button("📥 Télécharger analyses faciales", csv, "analyses_faciales.csv")
            if not ocr_df.empty:
                csv = ocr_df.to_csv(index=False)
                st.download_button("📥 Télécharger analyses OCR", csv, "analyses_ocr.csv")

    with col_exp2:
        if st.button("🔄 Actualiser les données"):
            st.rerun()

# -----------------------------
# 🔚 Nettoyage et footer
# -----------------------------

# Nettoyage amélioré
try:
    # Nettoyer les fichiers temporaires à la fin
    cleanup_temp_files()

    if 'image_path' in locals() and image_path and os.path.exists(image_path):
        time.sleep(1)  # Attendre que les processus libèrent les fichiers
        try:
            os.unlink(image_path)
        except PermissionError:
            pass  # Ignorer si le fichier est encore utilisé

    if 'ref_path' in locals() and 'ref_path' in locals() and os.path.exists(ref_path):
        time.sleep(1)
        try:
            os.unlink(ref_path)
        except PermissionError:
            pass
except:
    pass

if 'webcam_processor' in st.session_state:
    st.session_state.webcam_processor.stop_webcam()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🔍 Système de Reconnaissance Faciale Avancé - Développé avec Streamlit</p>
    <p>Fonctionnalités: Base de données • Statistiques • Historique • Export des données</p>
</div>
""", unsafe_allow_html=True)