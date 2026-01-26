"""
Application RAG (Retrieval-Augmented Generation) pour la base documentaire ANSM
Optimisée pour CPU avec LlamaIndex, Streamlit, Ollama (gemma3:4b) et HuggingFace Embeddings
"""

import streamlit as st
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configuration des chemins
DOCUMENTS_PATH = Path(__file__).parent / "base_documentaire"
PERSIST_DIR = Path(__file__).parent / "storage"

# Prompt système strict pour limiter les réponses aux documents
SYSTEM_PROMPT = """Tu es un assistant expert spécialisé dans la base de données publique des médicaments (BDPM) de l'ANSM.

RÈGLES STRICTES À SUIVRE :
1. Tu dois UNIQUEMENT répondre en te basant sur les informations présentes dans les documents fournis.
2. Si l'information demandée n'est pas présente dans les documents, réponds clairement : "Je n'ai pas trouvé cette information dans la base documentaire."
3. Ne jamais inventer ou supposer des informations qui ne sont pas explicitement mentionnées dans les documents.
4. Cite les sources quand c'est pertinent (nom du document ou référence CIS).
5. Réponds en français de manière claire et structurée.
6. Si la question est ambiguë, demande des précisions.

Tu as accès aux documents suivants de la BDPM :
- CIS_bdpm.txt : Informations générales sur les médicaments (dénomination, forme, voie d'administration, statut AMM)
- CIS_CIP_bdpm.txt : Présentations des médicaments (code CIP, libellé, statut)
- CIS_COMPO_bdpm.txt : Composition des médicaments (substances actives, dosages)
- CIS_GENER_bdpm.txt : Groupes génériques
- CIS_HAS_SMR_bdpm.txt : Avis SMR de la HAS
- CIS_HAS_ASMR_bdpm.txt : Avis ASMR de la HAS
- CIS_CPD_bdpm.txt : Conditions de prescription et délivrance
- CIS_InfoImportantes : Informations importantes sur les médicaments
"""


@st.cache_resource
def load_embedding_model():
    """Charge le modèle d'embedding une seule fois."""
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder=str(Path(__file__).parent / "models_cache"),
    )


@st.cache_resource
def load_llm():
    """Charge le LLM Ollama une seule fois."""
    return Ollama(
        model="gemma3:4b",
        request_timeout=120.0,
        temperature=0.1,  # Température basse pour des réponses plus factuelles
    )


def configure_settings():
    """Configure les paramètres globaux de LlamaIndex."""
    Settings.embed_model = load_embedding_model()
    Settings.llm = load_llm()
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 200


@st.cache_resource
def load_or_create_index(_force_reload: bool = False):
    """
    Charge l'index depuis le stockage persistant ou le crée si nécessaire.
    Le paramètre _force_reload avec underscore est ignoré par le cache.
    """
    configure_settings()
    
    # Vérifier si un index persistant existe
    if PERSIST_DIR.exists() and not _force_reload:
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
            index = load_index_from_storage(storage_context)
            return index, "Index chargé depuis le stockage."
        except Exception as e:
            st.warning(f"Impossible de charger l'index existant: {e}. Création d'un nouvel index...")
    
    # Créer un nouvel index
    if not DOCUMENTS_PATH.exists():
        st.error(f"Le dossier {DOCUMENTS_PATH} n'existe pas!")
        return None, "Erreur: dossier de documents introuvable."
    
    # Charger les documents
    documents = SimpleDirectoryReader(
        input_dir=str(DOCUMENTS_PATH),
        filename_as_id=True,
        required_exts=[".txt"],
    ).load_data()
    
    if not documents:
        st.error("Aucun document .txt trouvé dans le dossier!")
        return None, "Erreur: aucun document trouvé."
    
    # Créer l'index vectoriel
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )
    
    # Persister l'index
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    
    return index, f"Index créé avec {len(documents)} documents."


def reindex_documents():
    """Force la réindexation des documents."""
    # Supprimer le cache de l'index
    load_or_create_index.clear()
    
    # Supprimer le stockage persistant
    if PERSIST_DIR.exists():
        import shutil
        shutil.rmtree(PERSIST_DIR)
    
    # Recréer l'index
    return load_or_create_index(_force_reload=True)


def get_query_engine(index):
    """Crée le moteur de requête avec le prompt système."""
    return index.as_query_engine(
        similarity_top_k=5,  # Nombre de chunks à récupérer
        streaming=False,
        system_prompt=SYSTEM_PROMPT,
    )


def main():
    """Fonction principale de l'application Streamlit."""
    
    # Configuration de la page
    st.set_page_config(
        page_title="RAG ANSM - Base des Médicaments",
        page_icon="💊",
        layout="wide",
    )
    
    st.title("💊 Assistant ANSM - Base des Médicaments")
    st.markdown("*Interrogez la base de données publique des médicaments (BDPM)*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.markdown("---")
        st.subheader("📁 Documents")
        st.info(f"Chemin: `{DOCUMENTS_PATH}`")
        
        # Bouton de réindexation
        if st.button("🔄 Réindexer les documents", type="primary", use_container_width=True):
            with st.spinner("Réindexation en cours... Cela peut prendre quelques minutes."):
                index, message = reindex_documents()
                if index:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        st.markdown("---")
        st.subheader("ℹ️ Informations")
        st.markdown("""
        **Modèles utilisés:**
        - 🤖 LLM: `gemma3:4b` (Ollama)
        - 📊 Embedding: `bge-small-en-v1.5`
        
        **Documents disponibles:**
        - Médicaments (CIS)
        - Présentations (CIP)
        - Compositions
        - Génériques
        - Avis HAS (SMR/ASMR)
        """)
    
    # Initialiser l'historique de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Charger l'index
    with st.spinner("Chargement de l'index et des modèles..."):
        index, status_message = load_or_create_index()
    
    if index is None:
        st.error("Impossible de charger ou créer l'index. Vérifiez la configuration.")
        return
    
    # Afficher le statut dans la sidebar
    with st.sidebar:
        st.success(status_message)
    
    # Créer le moteur de requête
    query_engine = get_query_engine(index)
    
    # Afficher l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Zone de saisie du chat
    if prompt := st.chat_input("Posez votre question sur les médicaments..."):
        # Ajouter le message utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Afficher le message utilisateur
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Générer la réponse
        with st.chat_message("assistant"):
            with st.spinner("🔍 Recherche dans la base documentaire..."):
                try:
                    response = query_engine.query(prompt)
                    response_text = str(response)
                    
                    # Afficher la réponse
                    st.markdown(response_text)
                    
                    # Afficher les sources (optionnel)
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        with st.expander("📚 Sources utilisées"):
                            for i, node in enumerate(response.source_nodes, 1):
                                score = getattr(node, 'score', 'N/A')
                                file_name = Path(node.node.metadata.get('file_path', 'Inconnu')).name
                                st.markdown(f"**Source {i}** (score: {score:.3f}): `{file_name}`")
                                st.text(node.node.text[:500] + "..." if len(node.node.text) > 500 else node.node.text)
                                st.markdown("---")
                    
                except Exception as e:
                    response_text = f"❌ Erreur lors de la génération de la réponse: {str(e)}"
                    st.error(response_text)
        
        # Ajouter la réponse à l'historique
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Message d'aide initial
    if not st.session_state.messages:
        st.info("""
        👋 **Bienvenue!** Posez vos questions sur les médicaments de la base ANSM.
        
        **Exemples de questions:**
        - "Quelle est la composition du Doliprane ?"
        - "Quels sont les génériques du paracétamol ?"
        - "Quel est l'avis SMR de ce médicament ?"
        - "Quelles sont les conditions de prescription ?"
        """)


if __name__ == "__main__":
    main()
